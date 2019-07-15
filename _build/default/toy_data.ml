open Owl

let dir = Cmdargs.(get_string "-d" |> force ~usage:"-d [dir]")
let in_dir = Printf.sprintf "%s/%s" dir

(* some parameters *)
let k = 500 (* number of trials *)

let n_sub = 25 (* number of neurons in each sub-pop *)

let n = 2 * n_sub (* total # of neurons *)

let dt = 1E-4 (* simulate time step *)

let sampling_dt = 2E-3 (* sampling resolution *)

let duration = 0.5 (* trial duration (s) *)

let tau = 20E-3 (* "membrane time constant" :) *)

(* build the "A matrix" of a "chain network"; to avoid build-up of variance down
   the chain (in the stochastic network), the weights decrease (bell-shape) as
   you go down *)
let chain ~n weight =
  let w =
    Mat.init_2d n n (fun i j ->
        if i = j + 1 then 1. +. (weight *. Maths.(exp (-.sqr (float i /. 5.)))) else 0.)
  in
  Mat.(w - eye n)


(* constructs the A matrix of a time-reversible system which achieves a desired
   spatial covariance *)
let reversible c = Mat.(0.5 $* neg (inv c))

(* build the system *)
let a, ell_noise =
  (* magic number that ends up determining the dominant time scale of
     fluctuations in the reversible system; see below; the very same factor
     should be applied to the input noise variance that goes into the reversible
     sub-system *)

  let factor = 3. in
  (* sequential (chain) sub-system; 0.2 is a good number, too :D *)

  let a_seq = chain ~n:n_sub 0.2 in
  (* find the controllability Gramian, ie stationary covariance matrix
     when the input is pure white noise -- which it will be *)
  let c_seq = Linalg.D.lyapunov a_seq Mat.(neg (eye n_sub)) in
  (* reversible sub-system that matches the Gramian, but scaled
     to speed up the fluctuations which would otherwise be super slow *)
  let a_rev = Mat.(factor $* reversible c_seq) in
  (* A matrix of the whole system -- note that I could well simulate the two
     systems separately (would be more efficient), but somehow it feels better
     to have a single net :) *)
  let a =
    let z = Mat.zeros n_sub n_sub in
    Mat.concat_vh [| [| a_seq; z |]; [| z; a_rev |] |]
  in
  (* cholesky factor of the input noise; just a diagonal matrix of 1 (for the
     first subsys) and sqrt(factor) for the second subsys *)
  let ell_noise =
    Mat.(sqrt (diagm (concat_horizontal (ones 1 n_sub) (factor $* ones 1 n_sub))))
  in
  a, ell_noise


(* overall Gramian *)
let c = Linalg.D.lyapunov a Mat.(neg (ell_noise *@ transpose ell_noise))

(* simulate the dynamics *)
let simulate () =
  (* draw initial conditions (NxK) from the stationary distribution already *)
  let x0 =
    let ell = Linalg.D.chol ~upper:false c in
    Mat.(ell *@ gaussian n k)
  in
  let n_bins = int_of_float (duration /. dt) in
  let sample_every = int_of_float (sampling_dt /. dt) in
  (* iterate over time *)
  let rec iter x t accu =
    Printf.printf "\rt=%05i%!" t;
    if t mod 10 = 0 then Gc.minor ();
    if t = n_bins
    then
      (* if we are done, return the data in the right format *)
      accu
      |> List.rev
      |> Array.of_list
      |> Array.map (fun m -> Arr.expand m 3)
      |> Arr.concatenate ~axis:0 (* T x N x K *)
      |> Arr.transpose ~axis:[| 1; 0; 2 |]
      (* N x T x K *)
    else (
      (* only record the activity every [sample_every] time bins *)
      let accu = if t mod sample_every = 0 then x :: accu else accu in
      (* draw the noise *)
      let noise = Mat.(Maths.(sqrt (dt /. tau)) $* ell_noise *@ gaussian n k) in
      (* update the state *)
      let x = Mat.(x + (dt /. tau $* a *@ x) + noise) in
      (* loop over to next time step *)
      iter x (t + 1) accu)
  in
  let data = iter x0 0 [] in
  (* scramble indices *)
  let ids =
    let z = Array.init n_sub (fun i -> i) |> Stats.shuffle |> Array.to_list in
    z @ List.map (( + ) n_sub) z
  in
  Arr.get_fancy [ L ids ] data


let () =
  let data = simulate () in
  Arr.save data (in_dir "toy_data.bin")

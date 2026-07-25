//! Deterministic hash-MAC implementation of the consensus crypto
//! interface, for simulation and unit tests.
//!
//! Keyed BLAKE3 sized into the BLS containers: constant-cost sign,
//! verify, and aggregate, so protocol-logic suites stop paying pairing
//! cost. The scheme **discriminates** — wrong signer, wrong message,
//! tampered signature, wrong signer set, and wrong order all fail
//! verification — so byzantine and fault-injection scenarios keep their
//! semantics. It is **not unforgeable**: a signature is recomputable
//! from the public key alone, which is sound only for a harness that
//! holds every key anyway and models byzantine behavior by fault
//! injection, not key secrecy. Never deploy outside tests.
//!
//! Aggregates are an ordered fold, stricter than BLS's commutative sum:
//! a call site that forgets to canonicalize to committee-index order
//! fails under mock only, which is the desired failure mode.

mod derive;
mod signer;
mod verifier;

pub use signer::MockSigner;
pub use verifier::MockVerifier;

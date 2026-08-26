//! The protocol hash behind the VM's hashing seam.
//!
//! Re-exported from the VM rather than defined twice: an envelope's
//! signing digest and every address the effect vocabulary derives go
//! through one hash, and two definitions would be two identities for one
//! value that drift exactly once. The VM needs it to answer an author
//! asking what a resource's address is, so that is where it lives.

pub use hyperscale_vm_types::ProtocolHasher;

#[cfg(test)]
mod tests {
    use hyperscale_hbor::hash::Hasher;

    use super::ProtocolHasher;

    /// The digest this repo has always taken, pinned as bytes.
    ///
    /// The definition moved into the VM, and a move is only safe if the
    /// answer did not: every address, every child key and every signing
    /// digest a network has agreed on is this function's output.
    #[test]
    fn the_protocol_hash_is_the_one_it_has_always_been() {
        assert_eq!(
            ProtocolHasher.hash(b"hyperscale/test", &[b"one", b"two"]).0,
            [
                0x02, 0xd3, 0xac, 0xe5, 0x44, 0x53, 0x0e, 0xba, 0xa5, 0x68, 0x41, 0x55, 0x9c, 0x53,
                0xa9, 0x9b, 0xab, 0x75, 0x22, 0xf7, 0xf1, 0xe3, 0xfd, 0x28, 0xe8, 0x06, 0x65, 0x85,
                0xe9, 0x57, 0x9e, 0xed,
            ]
        );
    }
}

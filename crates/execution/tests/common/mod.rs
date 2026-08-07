//! Shared test scaffolding for the execution crate's integration tests.
//!
//! Each test binary compiles this module separately and uses only the part
//! it needs, so anything another binary drives reads as dead here.
#![allow(dead_code)]

pub mod sim;

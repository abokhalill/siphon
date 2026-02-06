//! Phase-A Frontend: Protocol Parsing and RIF Construction
//!
//! This module is part of the Trusted Computing Base (TCB).
//! All parsing and RIF construction happens here.

mod parser;
mod protocol;

pub use parser::{parse_protocol, ParseError};
pub use protocol::{Protocol, ProtocolField, FieldConstraint};

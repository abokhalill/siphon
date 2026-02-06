//! Protocol parser for Phase-A
//!
//! Parses protocol definitions into Protocol structs.

use crate::rif::ScalarType;
use super::protocol::{Protocol, ProtocolField, FieldConstraint};

fn parse_int_literal(s: &str) -> Option<u64> {
    let s = s.trim();
    if s.starts_with("0x") || s.starts_with("0X") {
        u64::from_str_radix(&s[2..], 16).ok()
    } else if s.starts_with("0b") || s.starts_with("0B") {
        u64::from_str_radix(&s[2..], 2).ok()
    } else if s.starts_with("0o") || s.starts_with("0O") {
        u64::from_str_radix(&s[2..], 8).ok()
    } else {
        s.parse().ok()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParseError {
    EmptyInput,
    InvalidFieldDeclaration { line: usize, reason: &'static str },
    InvalidType { line: usize, type_name: String },
    InvalidOffset { line: usize },
    InvalidConstraint { line: usize, reason: &'static str },
    DuplicateFieldName { line: usize, name: String },
    OverlappingFields { line: usize, field1: String, field2: String },
    FieldExceedsMaxSize { line: usize, field: String, end_offset: u32, max_size: u16 },
}

pub fn parse_protocol(content: &str) -> Result<Protocol, ParseError> {
    if content.trim().is_empty() {
        return Err(ParseError::EmptyInput);
    }

    let mut name = String::from("unnamed");
    let mut version: u8 = 1;
    let mut fields = Vec::new();
    let mut current_offset: u32 = 0;
    let mut max_size: u16 = 64;
    let mut field_names = std::collections::HashSet::new();

    for (line_num, line) in content.lines().enumerate() {
        let line = line.trim();

        if line.is_empty() || line.starts_with("//") || line.starts_with('#') {
            continue;
        }

        if line.starts_with("protocol ") {
            name = line
                .strip_prefix("protocol ")
                .unwrap_or("unnamed")
                .trim_end_matches('{')
                .trim()
                .to_string();
            continue;
        }

        if line.starts_with("version:") {
            let v = line
                .strip_prefix("version:")
                .unwrap_or("1")
                .trim()
                .trim_end_matches(';');
            version = v.parse().unwrap_or(1);
            continue;
        }

        if line.starts_with("max_size:") {
            let s = line
                .strip_prefix("max_size:")
                .unwrap_or("64")
                .trim()
                .trim_end_matches(';');
            max_size = s.parse().unwrap_or(64);
            continue;
        }

        if line.contains(':') && !line.starts_with("version") && !line.starts_with("max_size") {
            let field = parse_field(line, &mut current_offset, line_num, max_size)?;

            if field_names.contains(&field.name) {
                return Err(ParseError::DuplicateFieldName {
                    line: line_num + 1,
                    name: field.name,
                });
            }
            field_names.insert(field.name.clone());

            fields.push(field);
        }
    }

    Ok(Protocol {
        name,
        version,
        fields,
        max_size,
    })
}

fn parse_field(
    line: &str,
    current_offset: &mut u32,
    line_num: usize,
    max_size: u16,
) -> Result<ProtocolField, ParseError> {
    let line = line.trim().trim_end_matches(';').trim_end_matches(',');

    let parts: Vec<&str> = line.splitn(2, ':').collect();
    if parts.len() < 2 {
        return Err(ParseError::InvalidFieldDeclaration {
            line: line_num + 1,
            reason: "missing colon separator",
        });
    }

    let name = parts[0].trim().to_string();
    if name.is_empty() {
        return Err(ParseError::InvalidFieldDeclaration {
            line: line_num + 1,
            reason: "empty field name",
        });
    }

    let rest = parts[1].trim();
    let tokens: Vec<&str> = rest.split_whitespace().collect();
    if tokens.is_empty() {
        return Err(ParseError::InvalidFieldDeclaration {
            line: line_num + 1,
            reason: "missing type",
        });
    }

    let scalar_type = match tokens[0] {
        "u8" => ScalarType::U8,
        "u16" => ScalarType::U16,
        "u32" => ScalarType::U32,
        "u64" => ScalarType::U64,
        "i32" => ScalarType::I32,
        "i64" => ScalarType::I64,
        "f32" => ScalarType::F32,
        "f64" => ScalarType::F64,
        other => {
            return Err(ParseError::InvalidType {
                line: line_num + 1,
                type_name: other.to_string(),
            });
        }
    };

    let mut offset = *current_offset;
    let mut constraint = None;

    for token in &tokens[1..] {
        if token.starts_with("@offset(") {
            let val = token
                .strip_prefix("@offset(")
                .and_then(|s| s.strip_suffix(')'))
                .and_then(|s| s.parse().ok());
            match val {
                Some(o) => offset = o,
                None => {
                    return Err(ParseError::InvalidOffset { line: line_num + 1 });
                }
            }
        } else if token.starts_with("@range(") {
            let inner = token
                .strip_prefix("@range(")
                .and_then(|s| s.strip_suffix(')'));
            match inner {
                Some(range_str) => {
                    let bounds: Vec<&str> = range_str.split(',').collect();
                    if bounds.len() == 2 {
                        let lo = bounds[0].trim().parse().unwrap_or(0);
                        let hi = bounds[1].trim().parse().unwrap_or(u64::MAX);
                        if lo > hi {
                            return Err(ParseError::InvalidConstraint {
                                line: line_num + 1,
                                reason: "range lo > hi",
                            });
                        }
                        constraint = Some(FieldConstraint::Range { lo, hi });
                    } else {
                        return Err(ParseError::InvalidConstraint {
                            line: line_num + 1,
                            reason: "range requires two values",
                        });
                    }
                }
                None => {
                    return Err(ParseError::InvalidConstraint {
                        line: line_num + 1,
                        reason: "malformed @range",
                    });
                }
            }
        } else if token.starts_with("@equals(") {
            let val_str = token
                .strip_prefix("@equals(")
                .and_then(|s| s.strip_suffix(')'));
            match val_str.and_then(parse_int_literal) {
                Some(v) => constraint = Some(FieldConstraint::Equals(v)),
                None => {
                    return Err(ParseError::InvalidConstraint {
                        line: line_num + 1,
                        reason: "invalid @equals value",
                    });
                }
            }
        } else if *token == "@nonzero" {
            constraint = Some(FieldConstraint::NonZero);
        }
    }

    let field_end = offset + scalar_type.size_bytes() as u32;
    if field_end > max_size as u32 {
        return Err(ParseError::FieldExceedsMaxSize {
            line: line_num + 1,
            field: name,
            end_offset: field_end,
            max_size,
        });
    }

    *current_offset = field_end;

    Ok(ProtocolField {
        name,
        offset,
        scalar_type,
        constraint,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_protocol() {
        let input = r#"
protocol SimplePacket {
    version: 1;
    max_size: 64;

    magic: u32 @offset(0) @equals(0xDEADBEEF)
    length: u16 @range(1,1500)
    flags: u8 @nonzero
}
"#;
        let proto = parse_protocol(input).unwrap();
        assert_eq!(proto.name, "SimplePacket");
        assert_eq!(proto.fields.len(), 3);
        assert_eq!(proto.fields[0].name, "magic");
        assert_eq!(proto.fields[0].offset, 0);
        assert!(matches!(proto.fields[0].constraint, Some(FieldConstraint::Equals(0xDEADBEEF))));
    }

    #[test]
    fn test_parse_empty_input() {
        let result = parse_protocol("");
        assert!(matches!(result, Err(ParseError::EmptyInput)));
    }

    #[test]
    fn test_parse_invalid_type() {
        let input = "field: invalid_type";
        let result = parse_protocol(input);
        assert!(matches!(result, Err(ParseError::InvalidType { .. })));
    }

    #[test]
    fn test_parse_duplicate_field() {
        let input = r#"
max_size: 64;
field: u32
field: u32
"#;
        let result = parse_protocol(input);
        assert!(matches!(result, Err(ParseError::DuplicateFieldName { .. })));
    }

    #[test]
    fn test_parse_field_exceeds_max() {
        let input = r#"
max_size: 8;
big_field: u64 @offset(4)
"#;
        let result = parse_protocol(input);
        assert!(matches!(result, Err(ParseError::FieldExceedsMaxSize { .. })));
    }

    #[test]
    fn test_parse_invalid_range() {
        let input = r#"
max_size: 64;
field: u32 @range(100,50)
"#;
        let result = parse_protocol(input);
        assert!(matches!(result, Err(ParseError::InvalidConstraint { .. })));
    }
}

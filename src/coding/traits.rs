use binius_field::Field as BiniusField;
use thiserror::Error;

/// Error type for encoding and decoding operations
#[derive(Error, Debug, PartialEq)]
pub enum CodingError {
    #[error("Invalid parameters provided")]
    InvalidParameters,

    #[error("Insufficient data for decoding")]
    InsufficientData,

    #[error("Invalid symbol size")]
    InvalidSymbolSize,

    #[error("Encoding failed")]
    EncodingFailed,

    #[error("Decoding failed")]
    DecodingFailed,

    #[error("Invalid packet format")]
    InvalidPacketFormat,

    #[error("Not configured")]
    NotConfigured,

    #[error("No data set")]
    NoDataSet,

    #[error("Invalid data size")]
    InvalidDataSize,

    #[error("Invalid coefficients")]
    InvalidCoefficients,

    #[error("Redundant contribution - does not increase matrix rank")]
    RedundantContribution,
}

/// Trait for network encoders
pub trait Encoder<F: BiniusField> {
    /// Configure the encoder with parameters
    fn configure(&mut self, symbols: usize, symbol_size: usize) -> Result<(), CodingError>;

    /// Set the data to be encoded
    fn set_data(&mut self, data: &[u8]) -> Result<(), CodingError>;

    /// Generate an encoded symbol
    fn encode_symbol(&mut self, coefficients: &[F]) -> Result<Vec<u8>, CodingError>;

    /// Generate a coded packet with coefficients
    fn encode_packet(&mut self) -> Result<(Vec<F>, Vec<u8>), CodingError>;

    /// Get the number of source symbols
    fn symbols(&self) -> usize;

    /// Get the symbol size in bytes
    fn symbol_size(&self) -> usize;
}

/// Trait for network decoders
pub trait Decoder<F: BiniusField> {
    /// Configure the decoder with parameters
    fn configure(&mut self, symbols: usize, symbol_size: usize) -> Result<(), CodingError>;

    /// Add a received coded symbol
    fn add_symbol(&mut self, coefficients: &[F], symbol: &[u8]) -> Result<(), CodingError>;

    /// Check if decoding is possible
    fn can_decode(&self) -> bool;

    /// Attempt to decode the original data
    fn decode(&mut self) -> Result<Vec<u8>, CodingError>;

    /// Get the number of symbols needed for decoding
    fn symbols_needed(&self) -> usize;

    /// Get the number of symbols received so far
    fn symbols_received(&self) -> usize;

    /// Get the symbol size in bytes
    fn symbol_size(&self) -> usize;
}

/// Trait for streaming network decoders with incremental capabilities
pub trait StreamingDecoder<F: BiniusField>: Decoder<F> {
    /// Get the current rank of the decoding matrix
    fn current_rank(&self) -> usize;

    /// Check if a specific source symbol has been decoded
    fn is_symbol_decoded(&self, index: usize) -> bool;

    /// Get the number of symbols that have been fully decoded
    fn symbols_decoded(&self) -> usize;

    /// Attempt to decode a specific symbol without full decoding
    fn decode_symbol(&mut self, index: usize) -> Result<Option<Vec<u8>>, CodingError>;

    /// Check if new coefficients would increase the matrix rank
    fn check_rank_increase(&self, coefficients: &[F]) -> bool;

    /// Get the current decoding progress as a fraction
    fn progress(&self) -> f64 {
        let total = self.symbols_received() + self.symbols_needed();
        if total == 0 {
            0.0
        } else {
            self.symbols_received() as f64 / total as f64
        }
    }
}

/// Trait for network decoders that support recoding/relay functionality
pub trait RecodingDecoder<F: BiniusField>: Decoder<F> {
    /// Generate a recoded symbol from received symbols
    /// This allows the decoder to act as a relay node
    fn recode(&mut self, recode_coefficients: &[F]) -> Result<Vec<u8>, CodingError>;

    /// Check if the decoder has enough symbols to act as a recoder
    fn can_recode(&self) -> bool;

    /// Get the number of symbols needed for recoding (typically same as decoding)
    fn symbols_needed_for_recode(&self) -> usize;
}

/// Trait for streaming network encoders
pub trait StreamingEncoder<F: BiniusField>: Encoder<F> {
    /// Generate a specific encoded symbol by index (for systematic codes)
    fn encode_symbol_at(&mut self, index: usize) -> Result<Vec<u8>, CodingError>;

    /// Get the current encoding position (for carousel/streaming)
    fn current_position(&self) -> usize;

    /// Reset encoding position to start
    fn reset_position(&mut self);

    /// Check if all source symbols have been transmitted
    fn is_complete(&self) -> bool;
}

use binius_field::Field as BiniusField;
use thiserror::Error;

/// Error type for encoding and decoding operations
#[derive(Error, Debug, PartialEq)]
pub enum CodingError {
    /// Invalid parameters were provided to the encoder or decoder
    #[error("Invalid parameters provided")]
    InvalidParameters,

    /// Insufficient data has been received for successful decoding
    #[error("Insufficient data for decoding")]
    InsufficientData,

    /// The symbol size is invalid (e.g., zero or too large)
    #[error("Invalid symbol size")]
    InvalidSymbolSize,

    /// The encoding operation failed due to internal error
    #[error("Encoding failed")]
    EncodingFailed,

    /// The decoding operation failed due to insufficient or invalid data
    #[error("Decoding failed")]
    DecodingFailed,

    /// The packet format is invalid or corrupted
    #[error("Invalid packet format")]
    InvalidPacketFormat,

    /// The encoder or decoder has not been properly configured
    #[error("Not configured")]
    NotConfigured,

    /// No data has been set for encoding
    #[error("No data set")]
    NoDataSet,

    /// The provided data size is invalid for the current configuration
    #[error("Invalid data size")]
    InvalidDataSize,

    /// The provided coefficients are invalid (e.g., all zeros)
    #[error("Invalid coefficients")]
    InvalidCoefficients,

    /// The symbol is redundant and does not increase the matrix rank
    #[error("Redundant contribution - does not increase matrix rank")]
    RedundantContribution,
}

/// Trait for network encoders
pub trait Encoder<F: BiniusField, const N: usize> {
    /// Configure the encoder with parameters
    fn configure(&mut self, symbols: usize) -> Result<(), CodingError>;

    /// Set the data to be encoded
    fn set_data(&mut self, data: &[u8]) -> Result<(), CodingError>;

    /// Generate an encoded symbol
    fn encode_symbol(
        &mut self,
        coefficients: &[F],
    ) -> Result<crate::storage::Symbol<N>, CodingError>;

    /// Generate a coded packet with coefficients
    fn encode_packet(&mut self) -> Result<(Vec<F>, crate::storage::Symbol<N>), CodingError>;

    /// Get the number of source symbols
    fn symbols(&self) -> usize;
}

/// Trait for network decoders
pub trait Decoder<F: BiniusField, const N: usize> {
    /// Configure the decoder with parameters
    fn configure(&mut self, symbols: usize) -> Result<(), CodingError>;

    /// Add a received coded symbol
    fn add_symbol(
        &mut self,
        coefficients: &[F],
        symbol: &crate::storage::Symbol<N>,
    ) -> Result<(), CodingError>;

    /// Check if decoding is possible
    fn can_decode(&self) -> bool;

    /// Attempt to decode the original data
    fn decode(&mut self) -> Result<Vec<u8>, CodingError>;

    /// Get the number of symbols needed for decoding
    fn symbols_needed(&self) -> usize;

    /// Get the number of symbols received so far
    fn symbols_received(&self) -> usize;
}

/// Trait for streaming network decoders with incremental capabilities
pub trait StreamingDecoder<F: BiniusField, const N: usize>: Decoder<F, N> {
    /// Get the current rank of the decoding matrix
    fn current_rank(&self) -> usize;

    /// Check if a specific source symbol has been decoded
    fn is_symbol_decoded(&self, index: usize) -> bool;

    /// Get the number of symbols that have been fully decoded
    fn symbols_decoded(&self) -> usize;

    /// Attempt to decode a specific symbol without full decoding
    fn decode_symbol(
        &mut self,
        index: usize,
    ) -> Result<Option<crate::storage::Symbol<N>>, CodingError>;

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
pub trait RecodingDecoder<F: BiniusField, const N: usize>: Decoder<F, N> {
    /// Generate a recoded symbol from received symbols
    /// This allows the decoder to act as a relay node
    fn recode(
        &mut self,
        recode_coefficients: &[F],
    ) -> Result<crate::storage::Symbol<N>, CodingError>;

    /// Check if the decoder has enough symbols to act as a recoder
    fn can_recode(&self) -> bool;

    /// Get the number of symbols needed for recoding (typically same as decoding)
    fn symbols_needed_for_recode(&self) -> usize;
}

/// Trait for streaming network encoders
pub trait StreamingEncoder<F: BiniusField, const N: usize>: Encoder<F, N> {
    /// Generate a specific encoded symbol by index (for systematic codes)
    fn encode_symbol_at(&mut self, index: usize) -> Result<crate::storage::Symbol<N>, CodingError>;

    /// Get the current encoding position (for carousel/streaming)
    fn current_position(&self) -> usize;

    /// Reset encoding position to start
    fn reset_position(&mut self);

    /// Check if all source symbols have been transmitted
    fn is_complete(&self) -> bool;
}

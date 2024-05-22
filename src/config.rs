use serde::{Serialize, Deserialize};

pub type Int = u8;
pub type Float = f32;
pub const EPSILON: Float = f32::EPSILON;

pub trait ToFloat {
    fn as_float(&self) -> Float;
}

impl ToFloat for Int {
    fn as_float(&self) -> Float {
        *self as Float
    }
}

impl ToFloat for Float {
    fn as_float(&self) -> Float {
        *self
    }
}

#[derive(Deserialize, Serialize, Clone)]
pub struct TreeConfig {

    pub index_cols: Vec<String>,
    pub feature_cols: Vec<String>,
    pub cat_cols: Vec<String>,
    pub treatment_cols: Vec<String>,
    pub y_col: String,
    pub weight_col: String,

    pub n_bin: usize,
    pub min_samples_leaf: usize,
    pub min_samples_treatment: usize,
    pub n_reg: usize,
    pub alpha: Float,
    pub normalization: bool,

    pub max_features: usize,
    pub max_depth: usize,
    pub n_tree: usize,
    pub subsample: Float,
    pub n_thread: usize,
    pub seed: Option<u64>
}

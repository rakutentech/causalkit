use crate::config::{Int, Float};
use serde::{Serialize, Deserialize};

pub trait SplitInfoTrait {
    fn new() -> Self;
    fn set_node_id(&mut self, node_id: usize);
    fn set_feature_id(&mut self, feature_id: usize);
    fn set_treatment_id(&mut self, treatment_id: usize);
    fn set_iscat(&mut self, iscat: bool);
    fn set_value(&mut self, value: Option<Vec<Int>>);
    fn set_gain(&mut self, gain: Float);
    fn set_gain_importance(&mut self, gain_importance: Float);
    fn set_summary(&mut self, summary: Vec<Vec<Float>>);
    fn get_node_id(& self) -> usize;
    fn get_feature_id(& self) -> usize;
    fn get_treatment_id(& self) -> usize;
    fn get_iscat(& self) -> bool;
    fn get_value(& self) -> Option<Vec<Int>>;
    fn get_gain(& self) -> Float;
    fn get_gain_importance(& self) -> Float;
    fn get_summary(& self) -> Vec<Vec<Float>>;
}

#[derive(Clone, Deserialize, Serialize, Default)]
pub struct SplitInfo {
    pub node_id: usize,
    pub feature_id: usize,
    pub treatment_id: usize,
    pub iscat: bool,
    #[serde(default)]
    pub value: Option<Vec<Int>>,
    pub gain: Float,
    pub gain_importance: Float,
    pub summary: Vec<Vec<Float>>
}

impl SplitInfoTrait for SplitInfo {

    fn new() -> Self {
        SplitInfo::default()
    }

    fn set_node_id(&mut self, node_id: usize) {
        self.node_id = node_id;
    }

    fn set_feature_id(&mut self, feature_id: usize) {
        self.feature_id = feature_id;
    }

    fn set_treatment_id(&mut self, treatment_id: usize) {
        self.treatment_id = treatment_id;
    }

    fn set_iscat(&mut self, iscat: bool) {
        self.iscat = iscat;
    }

    fn set_value(&mut self, value: Option<Vec<Int>>) {
        self.value = value;
    }

    fn set_gain(&mut self, gain: Float) {
        self.gain = gain;
    }

    fn set_gain_importance(&mut self, gain_importance: Float) {
        self.gain_importance = gain_importance;
    }

    fn set_summary(&mut self, summary: Vec<Vec<Float>>) {
        self.summary = summary;
    }

    fn get_node_id(& self) -> usize {
        self.node_id
    }

    fn get_feature_id(& self) -> usize {
        self.feature_id
    }

    fn get_treatment_id(& self) -> usize {
        self.treatment_id
    }

    fn get_iscat(& self) -> bool {
        self.iscat
    }

    fn get_value(& self) -> Option<Vec<Int>> {
        match self.value.as_ref() {
            Some(value) => Some(value.clone()),
            None => None
        }
    }
    
    fn get_gain(& self) -> Float {
        self.gain
    }

    fn get_gain_importance(& self) -> Float {
        self.gain_importance
    }

    fn get_summary(& self) -> Vec<Vec<Float>> {
        self.summary.clone()
    }
}
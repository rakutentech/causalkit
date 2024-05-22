use std::collections::HashMap;

use crate::config::{Float, Int};

pub struct DMatrix {
    pub indices: Vec<Vec<String>>,
    pub feature: Vec<Vec<Option<Int>>>,
    pub response: Vec<Float>,
    pub treatments: Vec<Vec<Int>>,
    pub weights: Vec<Float>,
    pub is_bool: Vec<bool>,
    pub name: Vec<String>,
    pub bin_size: HashMap<String, usize>,
    pub treatment_size: Vec<usize>
}

impl DMatrix {

    pub fn new(indices: Vec<Vec<String>>, feature: Vec<Vec<Option<Int>>>, response: Vec<Float>, 
        treatments: Vec<Vec<Int>>, weights: Vec<Float>, is_bool: Vec<bool>, 
        name: Vec<String>, bin_size: HashMap<String, usize>, treatment_size: Vec<usize>) -> DMatrix {

        DMatrix {
            indices,
            feature,
            response,
            treatments,
            weights,
            is_bool,
            name,
            bin_size,
            treatment_size
        }
    }

    pub fn size(& self) -> usize {
        self.feature.get(0).unwrap().len()
    }

    pub fn n_feature(& self) -> usize {
        self.feature.len()
    }
    
    pub fn get(& self, feature_idx: usize, index: usize) -> Option<Int> {
        self.feature[feature_idx][index]
    }

    pub fn sort_index(& self, feature_idx: usize, indices: &mut [usize]) {
        // make sure a, b are smaller than Int::MAX
        indices.sort_by(|a, b| {
            let av: Int = match self.feature[feature_idx][*a] {
                Some(v) => v,
                None => Int::MAX
            };

            let bv: Int = match self.feature[feature_idx][*b] {
                Some(v) => v,
                None => Int::MAX
            };

            av.partial_cmp(&bv).unwrap()
        });
    }
}

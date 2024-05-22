use std::cmp;
use std::default::Default;
use serde::{Serialize, Deserialize};

use crate::config::{Int, Float, EPSILON};


#[derive(Default, Clone, Debug, Deserialize, Serialize)]
pub struct ContinuousBin {
    pub n_bin: usize,
	pub threshold: Vec<Float>
}

impl ContinuousBin {

    pub fn new(n_bin: usize, threshold: Option<Vec<Float>>) -> ContinuousBin {
        let threshold = match threshold {
            Some(value) => value,
            None => Vec::new()
        };

        ContinuousBin {n_bin, threshold}
    }

    pub fn fit(&mut self, arr: &Vec<Option<Float>>) {
        let (indices, none_size) = ContinuousBin::sort(arr);

        let total_size = indices.len();
        let val_size = total_size - none_size;
        let bin_size = val_size / self.n_bin;

        let mut threshold: Vec<Float> = Vec::new();

        for i in 1..self.n_bin {
            let mut bd = i * bin_size;
            while bd < val_size {
                if bd == val_size - 1 {break;}
                let bd_next = bd + 1;

                let v = indices.get(bd).unwrap();
                let v = arr[*v].unwrap();

                let v1 = indices.get(bd_next).unwrap();
                let v1 = arr[*v1].unwrap();

                if v < v1 - EPSILON {
                    if threshold.len() == 0 || *threshold.last().unwrap() != v {
                        threshold.push(v);
                    }

                    break;
                }

                bd += 1;
            }
        }

        self.n_bin = threshold.len() + 1;
        self.threshold = threshold;
    }

    pub fn map(& self, arr: &Vec<Option<Float>>) -> Vec<Option<Int>> {
        arr.iter().map(|&v| self.discretize(v)).collect()
    }

    fn sort(arr: &Vec<Option<Float>>) -> (Vec<usize>, usize) {
        let n = arr.len();
        let mut indices: Vec<usize> = (0..n).collect();

        indices.sort_by(|a, b| {
            let av: Float = match arr[*a] {
                Some(v) => v,
                None => Float::INFINITY
            };

            let bv: Float = match arr[*b] {
                Some(v) => v,
                None => Float::INFINITY
            };

            av.partial_cmp(&bv).unwrap()
        });

        let mut count = 0 as usize;
        for idx in indices.iter() {
            match arr[*idx] {
                Some(_) => {},
                None => count += 1
            };
        }

        (indices, count)
    }

    fn discretize(& self, v: Option<Float>) -> Option<Int> {
        match v {
            Some(x) => {
                let count = self.threshold.iter().filter(|&val| *val <= x).count();
                Some(count as Int)
            },
            None => None
        }
    }
}

#[derive(Default, Clone, Debug, Deserialize, Serialize)]
pub struct DiscreteBin {
    pub n_bin: usize,
	pub mapping: Vec<Float>
}

impl DiscreteBin {

    pub fn new(n_bin: usize) -> DiscreteBin {
        let mapping: Vec<Float> = Vec::new();
        DiscreteBin { n_bin, mapping }
    }

    pub fn fit(&mut self, arr: &Vec<Option<Float>>) {
        let mut keys: Vec<Float> = Vec::new();
        let mut count: Vec<usize> = Vec::new();
        for v in arr.iter() {
            if v.is_none() {
                continue;
            }
            let w = v.unwrap();
            let pos = keys.iter().position(|&x| x == w);
            match pos {
                Some(x) => { count[x] = count[x] + 1;},
                None => {
                    keys.push(w);
                    count.push(1);
                }
            }
        }

        let mut freq: Vec<(&Float, &usize)> = keys.iter().zip(count.iter()).collect();
        freq.sort_by(|a, b| {
            let av: usize = *a.1;
            let bv: usize = *b.1;
            bv.partial_cmp(& av).unwrap()
        });

        let k = cmp::min(keys.len(), Int::MAX as usize - 1);
        let mapping: Vec<Float> = freq.iter().map(|(&key, _cnt)| key).take(k).collect();

        let mut n_bin = mapping.len();
        if keys.len() > mapping.len() {
            n_bin = n_bin + 1;
        }

        self.n_bin = n_bin;
        self.mapping = mapping;
    }

    pub fn map(& self, arr: &Vec<Option<Float>>) -> Vec<Option<Int>> {
        let not_found_index = self.mapping.len();

        arr.iter().map(|&v| {
            match v {
                Some(x) => {
                    let pos = self.mapping.iter().position(|&m| m == x);
                    let index = match pos {
                        Some(y) => y as Int,
                        None => not_found_index as Int
                    };
                    Some(index)
                },
                None => None
            }


        }).collect()
    }
}

use std::cmp;

use crate::split_info::SplitInfoTrait;
use crate::tree::Tree;
use crate::strategy::Strategy;
use crate::partition::Partition;
use crate::config::{Float, TreeConfig};
use crate::dmatrix::DMatrix;
use crate::random::Random;


#[derive(Clone)]
pub struct TreeLearn<T: Tree + Clone + std::marker::Send, S: Strategy + Clone + std::marker::Send> {
    pub max_features: usize,
    pub max_depth: usize,
    pub partition: Partition,
    pub tree: T,
    pub strategy: S,
    pub conf: TreeConfig
}

impl<T: Tree + Clone + std::marker::Send, S: Strategy + Clone + std::marker::Send> TreeLearn<T, S> {
    
    pub fn new(conf: &TreeConfig) -> TreeLearn<T, S> {
        let max_features = conf.max_features;
        let max_depth = conf.max_depth;
        let partition = Partition::new();
        let tree = T::new();
        let strategy = S::new(conf);
        let conf = conf.clone();

        let learn: TreeLearn<T, S> = TreeLearn {
            max_features,
            max_depth,
            partition,
            tree,
            strategy,
            conf
        };

        learn
    }

    pub fn from_string(ss: &String) -> TreeLearn<T, S> {
        let lines: Vec<&str> = ss.split("\n").collect();
        let conf: TreeConfig = serde_json::from_str(lines.get(0).unwrap()).unwrap();
        let ss = lines.get(1).unwrap().to_string();
        let tree = T::from_string(& ss);
        let mut learner: TreeLearn<T, S> = TreeLearn::new(& conf);

        learner.tree = tree;
        learner
    }

    pub fn to_string(& self) -> String {
        let mut line_conf = serde_json::to_string(& self.conf).unwrap();
        let line_tree = self.tree.to_string();

        line_conf.push_str("\n");
        line_conf.push_str(line_tree.as_str());

        line_conf
    }

    pub fn fit(& mut self, m: &DMatrix, conf: &TreeConfig, random: &mut Random) {
        let n = m.size();
        let subsample: Float = conf.subsample;
        let k = ((n as Float) * subsample) as usize;

        let ns = random.choose(n, k, true);
        self.partition = Partition::subsample(ns);

        self.grow_tree(0, m, conf, random);
    }

    pub fn grow_tree(& mut self, n: usize, m: &DMatrix, conf: &TreeConfig, random: &mut Random) {
        let split_info: S::T = self.find_best_split(n, m, random);
        let mut split_info_t = T::T::new();
        split_info_t.set_node_id(split_info.get_node_id());
        split_info_t.set_feature_id(split_info.get_feature_id());
        split_info_t.set_treatment_id(split_info.get_treatment_id());
        split_info_t.set_iscat(split_info.get_iscat());
        split_info_t.set_value(split_info.get_value());
        split_info_t.set_gain(split_info.get_gain());
        split_info_t.set_gain_importance(split_info.get_gain_importance());
        split_info_t.set_summary(split_info.get_summary());
        let split_info: T::T = split_info_t;

        self.tree.add_split(n, split_info.clone());
        if split_info.get_value().is_none(){
            return ()
        }

        if self.tree.depth(n) >= self.max_depth {
            return ()
        }

        let left_index = self.tree.add_left(n);
        let right_index = self.tree.add_right(n);

        let left_size = self.split(split_info.clone(), m);
        self.partition.split(split_info.get_node_id(), left_size);

        self.grow_tree(left_index, m, conf, random);
        self.grow_tree(right_index, m, conf, random);
    }

    fn find_best_split(&mut self, n: usize, m: &DMatrix, random: &mut Random) -> S::T {

        let n_feature = m.n_feature();
        let k = cmp::min(n_feature, self.max_features);
        let feature_ids = random.choose(n_feature, k, false);

        let mut best_split: Option<S::T> = None;

        for feature_id in feature_ids.iter() {
            let indices = self.partition.get_indices(&n);

            let split_info = self.strategy.find_best_split(m, n, *feature_id, indices);

            if best_split.is_none() {
                best_split = Some(split_info);
                continue;
            }

            if split_info.get_value().is_none() {
                continue;
            }

            if best_split.as_ref().unwrap().get_value().is_none() {
                best_split = Some(split_info);
                continue;
            }

            if split_info.get_gain() > best_split.as_ref().unwrap().get_gain() {
                best_split = Some(split_info);
            }
        }

        best_split.unwrap()
    }

    fn split(&mut self, split_info: T::T, m: &DMatrix) -> usize {
        let iscat = split_info.get_iscat();
        match iscat {
            x if x => self.split_cat(split_info, m),
            _ => self.split_cont(split_info, m)
        }
    }

    fn split_cont(&mut self, split_info: T::T, m: &DMatrix) -> usize {
        let n = split_info.get_node_id();
        let feature_id = split_info.get_feature_id();
        let indices = self.partition.get_indices(&n);
        m.sort_index(feature_id, indices);

        let split_value = split_info.get_value();
        let threshold = split_value.as_ref().unwrap().get(0).unwrap();

        let mut left_size = 0;
        for i in 0..indices.len() {
            let index = indices.get(i).unwrap();
            let value = m.get(feature_id, *index);

            if value.is_none() {
                break;
            }

            if value.unwrap() > *threshold {
                break;
            }

            left_size += 1;
        }

        left_size
    }

    fn split_cat(&mut self, split_info: T::T, m: &DMatrix) -> usize {
        let n = split_info.get_node_id();
        let feature_id = split_info.get_feature_id();
        let indices = self.partition.get_indices(&n);

        let split_value = split_info.get_value();
        let threshold = split_value.as_ref().unwrap().get(0).unwrap();

        let mut start = 0;
        let mut end = indices.len() - 1;

        while start < end {
            let left_idx = indices.get(start).unwrap();
            let left_value = m.get(feature_id, *left_idx);

            let right_idx = indices.get(end).unwrap();
            let right_value = m.get(feature_id, *right_idx);

            if (!left_value.is_none()) && (left_value.unwrap() == *threshold) {
                start = start + 1;
                continue;
            }

            if right_value.is_none() || (right_value.unwrap() != *threshold) {
                end = end - 1;
                continue;
            }

            let (x, y) = (indices[start], indices[end]);
            indices[start] = y;
            indices[end] = x;

            start = start + 1;
            end = end - 1;
        }

        start
    }

    pub fn predict(& mut self, m: &DMatrix) -> Vec<Vec<Float>> {
        let n = m.size();
        let n_nodes = self.tree.size();
        self.partition = Partition::refresh(n, n_nodes);
        self.recursive_split(0, m);

        let mut score: Vec<Vec<Float>> = Vec::new();
        for _ in 0..n {
            score.push(Vec::new());
        }

        for i in 0..n_nodes {
            if !self.tree.is_leaf(i) {
                continue;
            }

            let uplift = self.tree.get_uplift(i);
            let indices = self.partition.get_indices(&i);

            for k in indices.iter() {
                for f in uplift.iter() {
                    score[*k].push(f.clone());
                }
            }
        }
        score
    }

    pub fn recursive_split(&mut self, n: usize, m: &DMatrix) {
        let left_children = self.tree.get_left_children();
        let left_index = left_children.get(n).unwrap().clone();

        let right_children = self.tree.get_right_children();
        let right_index = right_children.get(n).unwrap().clone();

        if left_index.is_none() {
            return ()
        }

        let split_info = self.tree.get_split_info(n).unwrap();
        let left_size = self.split(split_info, m);

        let left_index = left_index.clone().unwrap();
        let right_index = right_index.clone().unwrap();

        let start = self.partition.start.get(n).unwrap().clone();
        let size = self.partition.size.get(n).unwrap().clone();
        let right_size = size - left_size;

        self.partition.set_node_range(left_index, start, left_size);
        self.partition.set_node_range(right_index, start + left_size, right_size);

        if left_size > 0 {
            self.recursive_split(left_index, m);
        }

        if right_size > 0 {
            self.recursive_split(right_index, m);
        }
    }
}

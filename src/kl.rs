use crate::split_info::SplitInfo;
use crate::config::{Int, EPSILON, TreeConfig};
use crate::strategy::Strategy;
use crate::config::Float;
use crate::statistic::Count;
use crate::dmatrix::DMatrix;
use crate::linalg;
use crate::linalg::Matrix;

#[derive(Clone)]
pub struct KLStrategy {
    min_samples_leaf: usize,
    min_samples_treatment: usize,
    n_reg: usize,
    alpha: Float,
    normalization: bool,
}

impl Strategy for KLStrategy {
    type T = SplitInfo;

    fn new(conf: &TreeConfig) -> Self {
        let min_samples_leaf = conf.min_samples_leaf;
        let min_samples_treatment = conf.min_samples_treatment;
        let n_reg = conf.n_reg;
        let alpha = conf.alpha;
        let normalization = conf.normalization;

        KLStrategy { min_samples_leaf, min_samples_treatment, n_reg, alpha, normalization }
    }

    fn find_best_split(& self, m: &DMatrix, node_id: usize, feature_id: usize, 
        indices: &mut [usize]) -> Self::T {

        let iscat = m.is_bool.get(feature_id).unwrap().clone();
        let name = m.name.get(feature_id).unwrap();
        let feature_size = m.bin_size.get(name).unwrap().clone();
        let treatment_id = 0;
        let treatment_size = m.treatment_size.get(treatment_id).unwrap().clone();

        let hist = Count::calculate(m, feature_id, iscat, treatment_id, indices,
            feature_size, treatment_size);

        let (value, gain, gain_importance, summary) = self.find_best_split_plain(&hist);

        SplitInfo { node_id, feature_id, treatment_id, iscat, value, gain, gain_importance, summary }
    }
}

impl KLStrategy {

    fn find_best_split_plain(& self, hist: &Count) -> (Option<Vec<Int>>, Float, Float, Vec<Vec<Float>>) {
        let mut best_gain = 0.0;
        let mut best_gain_importance = 0.0;
        let mut best_split_value = None;

        let stat = & hist.stat;
        let mut parent_count = Vec::new();
        for (_idx_y, arr2) in stat.iter().enumerate() {
            let mut cnt = Vec::new();
            for (_idx_t, arr1) in arr2.iter().enumerate() {
                let c = match hist.iscat {
                    x if x => {
                        let v: Vec<f32> = arr1.iter().map(|&x| x as f32).collect();
                        v.iter().sum::<f32>() as Float
                    },
                    _ => *arr1.last().unwrap()
                };

                cnt.push(c);
            }
            parent_count.push(cnt);
        }

        let parent = KLStrategy::count_reg(&parent_count, &None, self.min_samples_treatment, self.n_reg);
        let parent = Some(parent);
        let parent_score = KLStrategy::evaluation(parent.as_ref().unwrap());

        let n_splits = stat[0][0].len() - 1;

        for pos in 0..n_splits {
            let left_count = KLStrategy::get_count(&stat, pos);

            let matrix_parent = Matrix::new(&parent_count);
            let matrix_left = Matrix::new(&left_count);
            let matrix_right = matrix_parent.subtract(& matrix_left);
            let right_count = matrix_right.get_data();

            let result = self.calculate_gain(&left_count, right_count, &parent, 
                parent_score);

            if let Some(y) = result {
                let (gain, gain_importance) = y;
                if gain > best_gain {
                    best_gain = gain;
                    best_gain_importance = gain_importance;

                    let mut value = Vec::new();
                    value.push(pos as Int);                    
                    best_split_value = Some(value);
                }
            }
        }

        (best_split_value, best_gain, best_gain_importance, parent.unwrap())
    }

    fn get_count(stat: &Vec<Vec<Vec<Float>>>, pos: usize) -> Vec<Vec<Float>> {
        let mut count = Vec::new();
        for (_idx_y, arr2) in stat.iter().enumerate() {
            let mut cnt = Vec::new();
            for (_idx_t, arr1) in arr2.iter().enumerate() {
                cnt.push(arr1[pos]);
            }
            count.push(cnt);
        }

        count
    }

    /*
    return [[p, n]]    p: prob of positive; n: total count
    */
    fn count_reg(count: &Vec<Vec<Float>>, parent: &Option<Vec<Vec<Float>>>, min_samples_treatment: usize, 
        n_reg: usize) -> Vec<Vec<Float>> {

        let mut node_summary = Vec::new();
        let n_groups = count.get(0).unwrap().len();

        for i in 0..n_groups {
            let n_neg: Float = count[0][i];
            let n_pos: Float = count[1][i];

            let n = n_neg + n_pos;

            let mut p = 0.0 as Float;
            if n > 0.0 as Float {
                p = n_pos / n
            }
            
            if !parent.is_none() {
                let pp = * parent.as_ref().unwrap().get(i).unwrap().get(0).unwrap();
                if n > min_samples_treatment as Float {
                    p = (n_pos + pp * n_reg as Float) / (n + n_reg as Float);
                } else {
                    p = pp;
                }
            }

            node_summary.push(vec![p, n]);
        }

        node_summary
    }

    fn evaluation(node_summary: &Vec<Vec<Float>>) -> Float {
        let p_c = node_summary[0][0];

        let n = node_summary.len();

        let mut s = 0.0;
        for i in 1..n {
            let p = node_summary[i][0];
            s = s + KLStrategy::kl_divergence(p, p_c);
        }

        s
    }

    /*
    pk: The probability of 1 in one distribution.
    qk: The probability of 1 in the other distribution.
    return KL divergence
    */
    fn kl_divergence(pk: Float, qk: Float) -> Float {
        let eps: Float = 1e-6;

        if qk < EPSILON {
            return 0.;
        }

        let qk_cap = linalg::float_min(linalg::float_max(qk, eps), 1.0 - eps);

        if pk < EPSILON {
            return -linalg::log(1.0 - qk_cap);
        }
        else if 1.0 - pk < EPSILON {
            return -linalg::log(qk_cap);
        } else {
            return pk * linalg::log(pk / qk_cap) + (1.0 - pk) * linalg::log((1.0 - pk) / (1.0 - qk_cap));
        }
    }
    
    fn calculate_gain(& self, left_count: &Vec<Vec<Float>>, right_count: &Vec<Vec<Float>>, 
        parent: &Option<Vec<Vec<Float>>>, parent_score: Float) -> Option<(Float, Float)> {

        let left_node_summary = KLStrategy::count_reg(left_count, parent, self.min_samples_treatment, self.n_reg);
        let right_node_summary = KLStrategy::count_reg(right_count, parent, self.min_samples_treatment, self.n_reg);

        let left_score = KLStrategy::evaluation(&left_node_summary);
        let right_score = KLStrategy::evaluation(&right_node_summary);

        let ln: Float = left_node_summary.iter().map(|x| *x.get(1).unwrap() as Float).sum();
        let rn: Float = right_node_summary.iter().map(|x| *x.get(1).unwrap() as Float).sum();

        let ln_min: Float = left_node_summary.iter().map(|x| *x.get(1).unwrap() as Float).min_by(
            |x, y| {
                if x.is_nan() {
                    std::cmp::Ordering::Less
                } else if y.is_nan() {
                    std::cmp::Ordering::Greater
                } else if x < y {
                    std::cmp::Ordering::Less
                } else if x > y {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Equal
                }
            }).unwrap();

        let rn_min: Float = right_node_summary.iter().map(|x| *x.get(1).unwrap() as Float).min_by(
            |x, y| {
                if x.is_nan() {
                    std::cmp::Ordering::Less
                } else if y.is_nan() {
                    std::cmp::Ordering::Greater
                } else if x < y {
                    std::cmp::Ordering::Less
                } else if x > y {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Equal
                }
            }).unwrap();

        if ln < self.min_samples_leaf as Float || rn < self.min_samples_leaf as Float {
            return None
        }

        if ln_min < self.min_samples_treatment as Float || rn_min < self.min_samples_treatment as Float {
            return None
        }

        let n: Float = ln + rn;
        let p  = ln / n;

        let mut gain = p * left_score + (1.0 - p) * right_score - parent_score;
        let gain_importance = ln * left_score + rn * right_score - n * parent_score;

        let norm_factor = match self.normalization {
            x if x => KLStrategy::norm(parent.as_ref().unwrap(), &left_node_summary, self.alpha),
            _ => 1.0
        };

        gain /= norm_factor;

        Some((gain, gain_importance))
    }

    fn norm(node_summary: &Vec<Vec<Float>>, left_node_summary: &Vec<Vec<Float>>, alpha: Float) -> Float {
        let mut norm_res = 0.0;

        let summary_parent = Matrix::new(node_summary);
        let summary_left = Matrix::new(left_node_summary);

        let mut n_tr = summary_parent.get_nth_column(1);
        let mut n_tr_left = summary_left.get_nth_column(1);

        let n_c = n_tr.remove(0);
        let n_c_left = n_tr_left.remove(0);

        let n_tr_sum: Float = n_tr.iter().sum();
        let n_tr_left_sum: Float = n_tr_left.iter().sum();

        let pt_a = n_tr_left_sum / (n_tr_sum + 0.1);
        let pc_a = n_c_left / (n_c + 0.1);

        // Normalization Part 1
        let v1 = alpha * KLStrategy::entropy_h(n_tr_sum / (n_tr_sum + n_c), n_c / (n_tr_sum + n_c)) * KLStrategy::kl_divergence(pt_a, pc_a);
        norm_res += v1;

        // Normalization Part 2 & 3
        let n_groups = n_tr.len();
        for i in 0..n_groups {
            let e_i = n_tr.get(i).unwrap();
            let le_i = n_tr_left.get(i).unwrap();

            let pt_a_i = le_i / (e_i + 0.1);
            let v2 = (1.0 - alpha) * KLStrategy::entropy_h(e_i / (e_i + n_c), n_c / (e_i + n_c)) * KLStrategy::kl_divergence(pt_a_i, pc_a);
            let v3 = e_i / (n_tr_sum + n_c) * KLStrategy::entropy_h(pt_a_i, -1.0);
            norm_res += v2;
            norm_res += v3;
        }

        // Normalization Part 4
        let v4 = n_c / (n_tr_sum + n_c) * KLStrategy::entropy_h(pc_a, -1.0);
        norm_res += v4;

        // Normalization Part 5
        norm_res += 0.5;

        norm_res
    }

    fn entropy_h(p: Float, q: Float) -> Float {
        if linalg::float_eq(q, -1.0) && p > 0.0 {
            return -p * linalg::log(p);
        } else if q > 0.0 {
            return -p * linalg::log(q);
        } else {
            return 0.0;
        }
    }
}

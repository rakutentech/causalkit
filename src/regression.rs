use crate::split_info::SplitInfo;
use crate::config::{Int, Float, TreeConfig};
use crate::strategy::Strategy;
use crate::dmatrix::DMatrix;
use crate::statistic::{Sum, SecondOrderSum, CountNoY};

#[derive(Clone)]
pub struct RegressionStrategy {
    pub min_samples_leaf: usize,
    pub min_samples_treatment: usize,
    pub alpha: Float
}

impl Strategy for RegressionStrategy {
    type T = SplitInfo;

    fn new(conf: &TreeConfig) -> Self {
        let min_samples_leaf = conf.min_samples_leaf;
        let min_samples_treatment = conf.min_samples_treatment;
        let alpha = conf.alpha;
        RegressionStrategy { min_samples_leaf, min_samples_treatment, alpha }
    }

    fn find_best_split(& self, m: &DMatrix, node_id: usize, feature_id: usize, 
        indices: &mut [usize]) -> Self::T {

        let iscat = m.is_bool.get(feature_id).unwrap().clone();
        let name = m.name.get(feature_id).unwrap();
        let feature_size = m.bin_size.get(name).unwrap().clone();
        let treatment_id = 0;
        let treatment_size = m.treatment_size.get(treatment_id).unwrap().clone();

        let sum = Sum::calculate(m, feature_id, iscat, treatment_id, indices,
            feature_size, treatment_size);

        let moment = SecondOrderSum::calculate(m, feature_id, iscat, treatment_id, indices,
            feature_size, treatment_size);

        let count = CountNoY::calculate(m, feature_id, iscat, treatment_id, indices,
            feature_size, treatment_size);

        let (value, gain, gain_importance) = self.find_best_split_plain(&sum, &moment, &count);

        let parent_sum = self.get_parent_stat(iscat, & sum.stat);
        let parent_count = self.get_parent_stat(iscat, & count.stat);

        let mut summary: Vec<Vec<Float>> = Vec::new();
        let n_treatment = parent_sum.len();
        for i in 0..n_treatment {
            let mut mean = Vec::new();
            mean.push(parent_sum[i] / parent_count[i]);
            summary.push(mean);
        }

        SplitInfo { node_id, feature_id, treatment_id, iscat, value, gain, gain_importance, summary }
    }
}

impl RegressionStrategy {
    fn find_best_split_plain(& self, sum: &Sum, moment: &SecondOrderSum, count: &CountNoY) -> 
        (Option<Vec<Int>>, Float, Float) {

        let mut best_gain = 0.0;
        let mut best_gain_importance = 0.0;
        let mut best_split_value = None;

        let iscat = sum.iscat;

        let parent_sum = self.get_parent_stat(iscat, & sum.stat);
        let parent_moment = self.get_parent_stat(iscat, & moment.stat);
        let parent_count = self.get_parent_stat(iscat, & count.stat);
        let parent_score = self.get_impurity(& parent_sum, & parent_moment, & parent_count);
        
        let n_splits = sum.stat[0].len() - 1;
        for pos in 0..n_splits {
            let left_sum = self.get_stat(& sum.stat, pos);
            let left_moment = self.get_stat(& moment.stat, pos);
            let left_count = self.get_stat(& count.stat, pos);
            let left_score = self.get_impurity(& left_sum, & left_moment, & left_count);

            let right_sum = self.get_difference(& parent_sum, & left_sum);
            let right_moment = self.get_difference(& parent_moment, & left_moment);
            let right_count = self.get_difference(& parent_count, & left_count);
            let right_score = self.get_impurity(& right_sum, & right_moment, & right_count);

            // https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
            // N_t / N is ignored since there is no pruning, comparison across nodes is not necessary
            // only support control and one treatment
            let n_t_l = left_count[0] + left_count[1];
            let n_t_r = right_count[0] + right_count[1];
            let n_t = parent_count[0] + parent_count[1];

            let gain = parent_score - left_score * n_t_l / n_t - right_score * n_t_r / n_t;

            if gain > best_gain {
                best_gain = gain;
                best_gain_importance = gain;

                let mut value = Vec::new();
                value.push(pos as Int);                    
                best_split_value = Some(value);
            }
        }
    
        (best_split_value, best_gain, best_gain_importance)
    }

    fn get_parent_stat(& self, iscat: bool, stat: &Vec<Vec<Float>>) -> Vec<Float> {
        let mut parent_stat = Vec::new();
        for (_idx_t, arr1) in stat.iter().enumerate() {
            let c = match iscat {
                x if x => {
                    let v: Vec<f32> = arr1.iter().map(|&x| x as f32).collect();
                    v.iter().sum::<f32>() as Float
                },
                _ => *arr1.last().unwrap()
            };

            parent_stat.push(c);
        }

        parent_stat
    }

    fn get_stat(& self, stat: &Vec<Vec<Float>>, pos: usize) -> Vec<Float> {
        let mut count = Vec::new();
        for (_idx_t, arr) in stat.iter().enumerate() {
            count.push(arr[pos]);
        }

        count
    }

    fn get_difference(& self, parent_stat: &Vec<Float>, child_stat: &Vec<Float>) -> Vec<Float> {
        let mut diff: Vec<Float> = Vec::new();

        let n = parent_stat.len();
        for pos in 0..n {
            diff.push(parent_stat[pos] - child_stat[pos]);
        }

        diff
    }

    fn get_impurity(& self, sum: &Vec<Float>, moment: &Vec<Float>, count: &Vec<Float>) -> Float {
        let tr_y_sum = sum[1];
        let tr_y_sq_sum = moment[1];
        let tr_count = count[1];
        let ct_y_sum = sum[0];
        let ct_y_sq_sum = moment[0];
        let ct_count = count[0];

        let tau = tr_y_sum / tr_count - ct_y_sum / ct_count;
        let tr_var = tr_y_sq_sum / tr_count - (tr_y_sum * tr_y_sum) / (tr_count * tr_count);
        let ct_var = ct_y_sq_sum / ct_count - (ct_y_sum * ct_y_sum) / (ct_count * ct_count);

        let impurity = (tr_var / tr_count + ct_var / ct_count) - tau * tau;
        let impurity = impurity + self.alpha * (tr_count - ct_count).abs();
        impurity
    }
}

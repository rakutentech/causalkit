use std::vec;
use std::error::Error;

use causality::config::{Float, TreeConfig};
use causality::rf::RandomForest;
use causality::kl::KLStrategy;
use causality::regression::RegressionStrategy;
use causality::tree::{ClassificationTree, RegressionTree};
use causality::util;

fn main() -> Result<(), Box<dyn Error>> {
    let model_type: String = util::parse_arg(1);
    let mode: String = util::parse_arg(2);
    let data_path: String = util::parse_arg(3);
    let model_path: String = util::parse_arg(4);
    let score_path: String = util::parse_arg(5);
    let y_name: String = util::parse_arg(6);
    let action_name: String = util::parse_arg(7);

    let index_cols: Vec<String> = Vec::new();

    let feature_cols_str: Vec<&str> = vec![
		"ｶｰﾄﾞｷｬｯｼﾝｸﾞ残高", "ｶｰﾄﾞｼｮｯﾋﾟﾝｸﾞ残高", "債権残高", "credit_limit",
		"枠消化率", "契約経過月数",
		"債務契約数", "ｾｸﾞﾒﾝﾄ分類",
		"初回請求ﾌﾗｸﾞ",
		"顧客ﾗｺCﾌﾗｸﾞ", "初使いﾗﾝｸ", "使い方ﾗﾝｸ", "初回楽天ﾗﾝｸ", "現在楽天ﾗﾝｸ",
		"年会費入金額", "過剰発生金額", "遅延損害金入金額", "その他入金額",
		"債権消込金額", "payment",
		"年収金額(万単位)",
		"年齢", "勤続年数",
		"性別区分", "職種ｺｰﾄﾞ", "住居状況ｺｰﾄﾞ", "業種ｺｰﾄﾞ",
		"正常顧客ﾌﾗｸﾞ", "顧客信用状態ﾌﾗｸﾞ",
		"ｼｮｯﾋﾟﾝｸﾞ利用可能額", "ｷｬｯｼﾝｸﾞ利用可能額", "ｼｮｯﾋﾟﾝｸﾞ利用金額", "ｷｬｯｼﾝｸﾞ利用金額", "ﾜﾝﾎﾟｲﾝﾄ利用金額",
		"過剰残高",
		"自振記録正常合計", "自振記録再振合計", "自振記録返戻合計", "過去1年間自振記録合計",
		"過去1年間未収有無", "過去1年間最大未収年令", "過去1年間未収月数",
		"取引目的-生活費ﾌﾗｸﾞ(1桁目)", "取引目的-融資ﾌﾗｸﾞ(2桁目)", "取引目的-事業費ﾌﾗｸﾞ(3桁目)", 
		"取引目的-学生ﾌﾗｸﾞ(4桁目)", "ｶｰﾄﾞ追加申込ﾌﾗｸﾞ",
		"ﾏﾈｰﾌﾗｸﾞ"];

    let mut feature_cols: Vec<String> = Vec::new();
    for col in feature_cols_str.iter() {
        feature_cols.push(col.to_string());
    }

    let cat_cols_str: Vec<&str> = vec![
		"債務契約数", "ｾｸﾞﾒﾝﾄ分類",
		"初回請求ﾌﾗｸﾞ",
		"顧客ﾗｺCﾌﾗｸﾞ", "初使いﾗﾝｸ", "使い方ﾗﾝｸ", "初回楽天ﾗﾝｸ", "現在楽天ﾗﾝｸ",
		"性別区分", "職種ｺｰﾄﾞ", "住居状況ｺｰﾄﾞ", "業種ｺｰﾄﾞ",
		"正常顧客ﾌﾗｸﾞ", "顧客信用状態ﾌﾗｸﾞ",
		"自振記録正常合計", "自振記録再振合計", "自振記録返戻合計", "過去1年間自振記録合計",
		"過去1年間未収有無", "過去1年間最大未収年令", "過去1年間未収月数",
		"取引目的-生活費ﾌﾗｸﾞ(1桁目)", "取引目的-融資ﾌﾗｸﾞ(2桁目)", "取引目的-事業費ﾌﾗｸﾞ(3桁目)", 
		"取引目的-学生ﾌﾗｸﾞ(4桁目)", "ｶｰﾄﾞ追加申込ﾌﾗｸﾞ",
		"ﾏﾈｰﾌﾗｸﾞ"];

    let mut cat_cols: Vec<String> = Vec::new();
    for col in cat_cols_str.iter() {
        cat_cols.push(col.to_string());
    }

    let treatment_cols: Vec<String> = vec![action_name];
    let y_col: String = y_name;
    let weight_col: String = "".to_string();

    let n_bin: usize = 30;
    let min_samples_leaf: usize = 100;
    let min_samples_treatment: usize = 10;
    let n_reg: usize = 10;
    let alpha: Float = 0.9;
    let normalization: bool = true;

    let max_features: usize = 20;
    let max_depth: usize = 6;
    let n_tree: usize = 30;
    let subsample: Float = 1.0;
    let n_thread: usize = 4;
    let seed: Option<u64> = Some(42);

    let conf = TreeConfig {
        index_cols, feature_cols, cat_cols, treatment_cols, y_col, weight_col,
        n_bin, min_samples_leaf, min_samples_treatment, n_reg, alpha, normalization,
        max_features, max_depth, n_tree, subsample, n_thread, seed
    };

    if mode == String::from("train") {
        if model_type == String::from("binary") {
            let mut model: RandomForest<ClassificationTree, KLStrategy> = RandomForest::new(conf);

            let m = model.loader.from_csv(data_path.clone());
            model.fit(m);
            model.save(& model_path)?;
        } else if model_type == String::from("reg") {
            let mut model: RandomForest<RegressionTree, RegressionStrategy> = RandomForest::new(conf);

            let m = model.loader.from_csv(data_path.clone());
            model.fit(m);
            model.save(& model_path)?;
        } else {
            panic! ("model_type not found");
        }
    }

    if mode == String::from("test") {
        if model_type == String::from("binary") {
            let mut model_predict = RandomForest::<ClassificationTree, KLStrategy>::load(& model_path).unwrap();

            let m = model_predict.loader.from_csv(data_path.clone());
            let score = model_predict.predict(m);
    
            util::write_csv(&score, &score_path)?;
        } else if model_type == String::from("reg") {
            let mut model_predict = RandomForest::<RegressionTree, RegressionStrategy>::load(& model_path).unwrap();

            let m = model_predict.loader.from_csv(data_path.clone());
            let score = model_predict.predict(m);
    
            util::write_csv(&score, &score_path)?;
        } else {
            panic! ("model_type not found");
        }
    }

    Ok(())
}

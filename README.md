# Driverless_AI_ClientAccess
Python and R client for H2O Driverless AI

---

## ドキュメント
- [Driverless AI Python Client](http://docs.h2o.ai/driverless-ai/pyclient/docs/html/index.html)

---
## Examples
### Non-Timeseries (IID Table)
- [IID_Table/PyClient_GettingStarted.ipynb](IID_Table/PyClient_GettingStarted.ipynb): まずはじめに（Driverless AI画面との比較付き実行例）
- [IID_Table/scoring.ipynb](IID_Table/scoring.ipynb): スコアリングの実施例
- [IID_Table/data_recipe.ipynb](IID_Table/data_recipe.ipynb): データレシピ（テキストのトークン化）の適用とスコアリング
- [IID_Table/ModelDiagnostics.ipynb](IID_Table/ModelDiagnostics.ipynb): Model Diagnostics

### Timeseries
- [Timeseries/DAI_PyClient_TS_example.ipynb](Timeseries/DAI_PyClient_TS_example.ipynb): Time Series Experimentの実施例
- [Timeseries/DAI_PyClient_TS_TTAorRefit.ipynb](Timeseries/DAI_PyClient_TS_TTAorRefit.ipynb): モデル運用 - ローリング予測（Refit or Test Time Augmentation(TTA)）の比較
- [Timeseries/TimeSeries_res_check.ipynb](Timeseries/TimeSeries_res_check.ipynb): Time Series Experimentの予測結果の確認(テストデータにおける結果をまとめて確認したい)

### Optimization
- [Optimization/GPyOpt_DAI_test.ipynb](Optimization/GPyOpt_DAI_test.ipynb): （検証コード）Driverless AIのスコアリング機能を目的関数とした、GPyOptによるベイズ最適化

### Management
- [Management/mlflow.ipynb](Management/mlflow.ipynb): MLFlowを用いたDriverless AIのExperimentの管理（基本操作）

### Experiment
- [Experiment/speed_test_result.ipynb](Experiment/speed_test_result.ipynb): CPUマシン vs GPUマシン

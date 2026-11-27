@echo off
setlocal
cd /d %~dp0

echo [1/8] Feature engineering...
python src\pipelines\07_feature_engineering.py || exit /b 1

echo [2/8] Feature selection...
python src\pipelines\08_feature_selection.py || exit /b 1

echo [3/8] Time split...
python src\pipelines\09_split_dataset.py || exit /b 1

echo [4/8] Train baseline models...
python src\pipelines\10_train_models.py || exit /b 1

echo [5/8] Tune models...
python src\pipelines\11_hyperparameter_tuning.py || exit /b 1

echo [6/8] Evaluate models and research experiments...
python src\pipelines\12_evaluate_models.py || exit /b 1
python src\pipelines\15_statistical_significance.py || exit /b 1
python src\pipelines\17_backtest_roi.py || exit /b 1
python src\pipelines\19_overfitting_analysis.py || exit /b 1
python src\pipelines\20_odds_ablation.py || exit /b 1
python src\pipelines\21_research_enhancements.py --calibration --ablation --walk-forward || exit /b 1

echo [7/8] Research audit...
python src\pipelines\22_research_audit.py || exit /b 1

echo [8/8] Tests...
pytest -q
if errorlevel 1 exit /b 1

echo.
echo RESEARCH UPGRADE COMPLETE.
pause

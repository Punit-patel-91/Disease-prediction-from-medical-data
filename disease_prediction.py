from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,ConfusionMatrixDisplay,RocCurveDisplay,classification_report

BASE=Path(__file__).resolve().parent; DATA=BASE/"data"; OUT=BASE/"outputs"; OUT.mkdir(exist_ok=True)
CSV_FILE="breast_cancer_dataset.csv"; TARGET="diagnosis"

def main():
    path=DATA/CSV_FILE
    if not path.exists(): raise FileNotFoundError(f"Dataset not found: {path}")
    df=pd.read_csv(path).dropna(subset=[TARGET]).drop_duplicates()
    if df[TARGET].nunique()!=2: raise ValueError("Target must contain exactly two classes.")
    X=df.drop(columns=[TARGET]); X=X.select_dtypes(include=["number","bool"])
    ytext=df[TARGET].astype(str); le=LabelEncoder(); y=le.fit_transform(ytext)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.20,random_state=42,stratify=y)
    def pipe(m): return Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",m)])
    models={
      "Logistic Regression":pipe(LogisticRegression(max_iter=3000,class_weight="balanced")),
      "SVM":pipe(SVC(kernel="rbf",probability=True,class_weight="balanced",random_state=42)),
      "Random Forest":Pipeline([("imputer",SimpleImputer(strategy="median")),("model",RandomForestClassifier(n_estimators=300,max_depth=10,min_samples_leaf=2,class_weight="balanced",random_state=42,n_jobs=-1))])
    }
    rows=[]; trained={}
    for name,m in models.items():
        m.fit(Xtr,ytr); pred=m.predict(Xte); prob=m.predict_proba(Xte)[:,1]
        rows.append({"Model":name,"Accuracy":accuracy_score(yte,pred),"Precision":precision_score(yte,pred,zero_division=0),"Recall":recall_score(yte,pred,zero_division=0),"F1-Score":f1_score(yte,pred,zero_division=0),"ROC-AUC":roc_auc_score(yte,prob)})
        trained[name]=(m,pred,prob)
    metrics=pd.DataFrame(rows); metrics.to_csv(OUT/"model_metrics.csv",index=False)
    best=metrics.sort_values("F1-Score",ascending=False).iloc[0]["Model"]; model,pred,prob=trained[best]
    fig,ax=plt.subplots(figsize=(6,5)); ConfusionMatrixDisplay(confusion_matrix(yte,pred),display_labels=le.classes_).plot(ax=ax,colorbar=False,cmap="Blues"); ax.set_title(f"Confusion Matrix - {best}"); plt.tight_layout(); plt.savefig(OUT/"01_confusion_matrix.png",dpi=180); plt.close()
    fig,ax=plt.subplots(figsize=(8,6))
    for name,(_,_,pr) in trained.items(): RocCurveDisplay.from_predictions(yte,pr,name=name,ax=ax)
    ax.set_title("ROC Curve Comparison"); plt.tight_layout(); plt.savefig(OUT/"02_roc_curve.png",dpi=180); plt.close()
    plt.figure(figsize=(7,5)); df[TARGET].value_counts().plot(kind="bar"); plt.title("Disease Class Distribution"); plt.xlabel("Class"); plt.ylabel("Samples"); plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(OUT/"03_class_distribution.png",dpi=180); plt.close()
    (OUT/"classification_report.txt").write_text(classification_report(yte,pred,target_names=le.classes_,zero_division=0),encoding="utf-8")
    joblib.dump(model,OUT/"best_model.joblib"); np.save(OUT/"class_names.npy",le.classes_)
    out=Xte.copy(); out["Actual"]=le.inverse_transform(yte); out["Predicted"]=le.inverse_transform(pred); out["Prediction_Probability"]=prob; out.to_csv(OUT/"test_predictions.csv",index=False)
    (OUT/"project_summary.txt").write_text(f"Dataset: {CSV_FILE}\nBest model by F1-Score: {best}\n\n{metrics.to_string(index=False)}\n\nAcademic use only; not a medical diagnostic tool.",encoding="utf-8")
    print(metrics.to_string(index=False)); print(f"\nBest model by F1-Score: {best}\nPROJECT COMPLETED SUCCESSFULLY")

if __name__=="__main__":
    try: main()
    except Exception as e: print("ERROR:",type(e).__name__,e)

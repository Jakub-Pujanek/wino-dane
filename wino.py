import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (auc, classification_report, confusion_matrix,
							 f1_score, precision_score, recall_score,
							 roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


warnings.filterwarnings("ignore")
st.set_page_config(page_title="Wine analytics and pairings", layout="wide")
sns.set_theme(style="whitegrid")


@st.cache_data(show_spinner=False)
def load_wine_quality() -> pd.DataFrame:
	"""Load and enrich the wine quality dataset."""

	df = pd.read_csv("winequality-red.csv")
	df = df.drop_duplicates().reset_index(drop=True)
	df["quality"] = df["quality"].astype(int)

	# Feature engineering to enrich downstream analysis.
	df["quality_label"] = pd.cut(
		df["quality"],
		bins=[0, 4, 6, 8, 10],
		labels=["Very low", "Low", "Medium", "High"],
		include_lowest=True,
	)

	df["high_quality_flag"] = (df["quality"] >= 6).astype(int)
	df["sulphates_to_acidity"] = df["sulphates"] / df["fixed acidity"].replace(0, np.nan)
	df["total_sulfur_ratio"] = df["free sulfur dioxide"] / df["total sulfur dioxide"].replace(
		0, np.nan
	)
	df["sugar_density_interaction"] = df["residual sugar"] * df["density"]
	df["alcohol_acidity_ratio"] = df["alcohol"] / df["fixed acidity"].replace(0, np.nan)

	return df


@st.cache_data(show_spinner=False)
def load_pairings() -> pd.DataFrame:
	"""Load the food pairing dataset with light cleansing."""

	df = pd.read_csv("wine_food_pairings.csv")
	df = df.drop_duplicates().reset_index(drop=True)
	df["pairing_quality"] = pd.to_numeric(df["pairing_quality"], errors="coerce")
	df["quality_label"] = df["quality_label"].fillna("Unknown")
	df["description"] = df["description"].fillna("")
	return df


def format_big_number(value: float) -> str:
	"""Compact formatting for metrics displayed in the UI."""

	if value is None or np.isnan(value):
		return "-"
	if abs(value) >= 1_000_000:
		return f"{value/1_000_000:.1f}M"
	if abs(value) >= 1_000:
		return f"{value/1_000:.1f}k"
	return f"{value:.0f}" if float(value).is_integer() else f"{value:.2f}"


def render_download_button(df: pd.DataFrame, label: str, file_name: str = "export.csv") -> None:
	"""Render a CSV download button for any filtered dataframe."""

	csv = df.to_csv(index=False).encode("utf-8")
	st.download_button(
		label=label,
		data=csv,
		file_name=file_name,
		mime="text/csv",
	)


def plot_correlation_heatmap(df: pd.DataFrame, cols: list[str]) -> go.Figure:
	corr = df[cols].corr().round(3)
	fig = go.Figure(
		data=go.Heatmap(
			z=corr.values,
			x=corr.columns,
			y=corr.index,
			colorscale="RdBu",
			reversescale=True,
			colorbar=dict(title="Correlation"),
			text=corr.values,
			texttemplate="%{text}",
		)
	)
	fig.update_layout(height=650, margin=dict(l=0, r=0, t=30, b=0))
	return fig


def plot_confusion_matrix_matrix(cm: np.ndarray, labels: list[str]) -> go.Figure:
	fig = go.Figure(
		data=go.Heatmap(
			z=cm,
			x=labels,
			y=labels,
			colorscale="Blues",
			showscale=False,
			text=cm,
			texttemplate="%{text}",
		)
	)
	fig.update_layout(
		width=400,
		height=400,
		margin=dict(l=20, r=20, t=40, b=20),
		xaxis=dict(title="Predicted"),
		yaxis=dict(title="Actual"),
	)
	return fig


def run_clustering_analysis(df: pd.DataFrame, n_clusters: int, random_state: int) -> tuple[pd.DataFrame, go.Figure, np.ndarray]:
	"""Execute k-means clustering and return enriched data, PCA plot, and variance."""

	numeric_cols = [c for c in df.columns if df[c].dtype != "O"]
	numeric_cols = [c for c in numeric_cols if c not in {"quality", "high_quality_flag"}]
	features = df[numeric_cols].fillna(df[numeric_cols].median())
	scaler = StandardScaler()
	scaled = scaler.fit_transform(features)

	kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
	clusters = kmeans.fit_predict(scaled)

	pca = PCA(n_components=2, random_state=random_state)
	components = pca.fit_transform(scaled)

	cluster_df = df.copy()
	cluster_df["cluster"] = clusters.astype(str)
	cluster_df["pc1"] = components[:, 0]
	cluster_df["pc2"] = components[:, 1]

	fig = px.scatter(
		cluster_df,
		x="pc1",
		y="pc2",
		color="cluster",
		hover_data=["quality", "alcohol", "residual sugar"],
		title="K-means (2D PCA)",
		color_discrete_sequence=px.colors.qualitative.Dark24,
	)
	fig.update_layout(height=600)

	return cluster_df, fig, pca.explained_variance_ratio_


def compute_top_terms(series: pd.Series, top_n: int = 20) -> pd.DataFrame:
	"""Extract the most frequent descriptive words from pairing notes."""

	tokens: list[str] = []
	stopwords = {
		"wine",
		"pairing",
		"pairs",
		"serve",
		"with",
		"food",
		"dish",
		"danie",
		"wino",
	}

	for text in series.fillna(""):
		normalized = (
			text.replace(",", " ")
			.replace(".", " ")
			.replace(";", " ")
			.replace("/", " ")
			.lower()
		)
		for token in normalized.split():
			if token.isalpha() and len(token) > 3 and token not in stopwords:
				tokens.append(token)

	counts = pd.Series(tokens).value_counts().head(top_n)
	return counts.reset_index().rename(columns={"index": "word", 0: "count"})


def render_word_frequency_bar(df_terms: pd.DataFrame) -> go.Figure:
	fig = px.bar(
		df_terms.sort_values("count"),
		x="count",
		y="word",
		orientation="h",
		title="Most frequent words",
		text="count",
		color="count",
		color_continuous_scale="Aggrnyl",
	)
	fig.update_layout(height=600, margin=dict(l=0, r=20, t=40, b=20))
	fig.update_traces(textposition="outside")
	return fig


def build_model(algorithm: str, random_state: int, num_class: int) -> object:
	if algorithm == "Logistic Regression":
		return LogisticRegression(max_iter=1000, random_state=random_state)
	if algorithm == "Random Forest":
		return RandomForestClassifier(n_estimators=400, random_state=random_state, n_jobs=-1)
	if algorithm == "XGBoost":
		params = {
			"objective": "binary:logistic" if num_class == 2 else "multi:softprob",
			"eval_metric": "logloss",
			"n_estimators": 400,
			"learning_rate": 0.05,
			"max_depth": 4,
			"subsample": 0.9,
			"colsample_bytree": 0.9,
			"reg_lambda": 1.0,
			"random_state": random_state,
			"verbosity": 0,
		}
		if num_class > 2:
			params["num_class"] = num_class
		return XGBClassifier(**params)
	raise ValueError("Unsupported algorithm")


def render_page_intro(df_quality: pd.DataFrame, df_pairings: pd.DataFrame) -> None:
	st.title("Comprehensive wine analytics")
	st.markdown(
		"""
		Use this Streamlit workspace to investigate wine quality metrics alongside
		detailed food pairing recommendations. Navigate with the sidebar to access
		data audits, interactive visualisations, predictive modelling, and
		cross-dataset insight boards.
		"""
	)

	col1, col2, col3, col4 = st.columns(4)
	col1.metric("Wine records", format_big_number(len(df_quality)))
	col2.metric("Average quality", f"{df_quality['quality'].mean():.2f}")
	col3.metric("Quality standard deviation", f"{df_quality['quality'].std():.2f}")
	col4.metric("Pairing records", format_big_number(len(df_pairings)))

	st.subheader("Data audit")
	qa_col1, qa_col2 = st.columns(2)
	with qa_col1:
		st.markdown("#### Wine Quality (numerics)")
		st.dataframe(df_quality.describe().T.round(3))
	with qa_col2:
		st.markdown("#### Wine Quality (missing values)")
		st.dataframe(df_quality.isna().sum().to_frame(name="missing"))

	st.markdown("#### Pairings - quick glance")
	pair_cols = st.multiselect(
		"Columns to preview",
		options=df_pairings.columns.tolist(),
		default=["wine_type", "wine_category", "food_item", "quality_label"],
	)
	st.dataframe(df_pairings[pair_cols].head(20))

	st.markdown("#### Wine extremes")
	extreme_col1, extreme_col2 = st.columns(2)
	with extreme_col1:
		st.markdown("**Top alcohol content**")
		st.dataframe(
			df_quality.nlargest(5, "alcohol")[
				["alcohol", "quality", "sulphates", "residual sugar"]
			]
		)
	with extreme_col2:
		st.markdown("**Lowest acidity (fixed)**")
		st.dataframe(
			df_quality.nsmallest(5, "fixed acidity")[
				["fixed acidity", "quality", "pH", "alcohol"]
			]
		)


def render_page_quality(df_quality: pd.DataFrame) -> None:
	st.header("Physicochemical exploration")

	st.markdown("### Core distributions")
	analytic_cols = df_quality.select_dtypes(include=[np.number]).columns.tolist()

	col_left, col_right = st.columns([2, 1])
	with col_left:
		default_index = analytic_cols.index("alcohol") if "alcohol" in analytic_cols else 0
		feature = st.selectbox("Select attribute", analytic_cols, index=default_index)
		fig_hist = px.histogram(
			df_quality,
			x=feature,
			color="quality_label",
			nbins=40,
			barmode="overlay",
			marginal="box",
			opacity=0.75,
			title=f"Distribution of {feature}",
		)
		st.plotly_chart(fig_hist, use_container_width=True)

	with col_right:
		st.markdown("#### Detailed statistics")
		stats = df_quality.groupby("quality_label")[feature].describe().round(2)
		st.dataframe(stats)

	st.markdown("### Quality overview")
	overview_col1, overview_col2 = st.columns(2)
	with overview_col1:
		nbins = max(1, len(df_quality["quality"].unique()))
		quality_fig = px.histogram(
			df_quality,
			x="quality",
			color="quality_label",
			title="Quality frequency",
			marginal="rug",
			nbins=nbins,
		)
		st.plotly_chart(quality_fig, use_container_width=True)
	with overview_col2:
		quality_corr = (
			df_quality.select_dtypes(include=[np.number]).corr()["quality"].drop("quality").dropna().sort_values()
		)
		corr_fig = px.bar(
			quality_corr.reset_index().rename(columns={"index": "feature", "quality": "correlation"}),
			x="correlation",
			y="feature",
			orientation="h",
			title="Correlation with quality",
			color="correlation",
			color_continuous_scale="RdBu",
		)
		corr_fig.update_layout(margin=dict(l=0, r=0, t=60, b=0))
		st.plotly_chart(corr_fig, use_container_width=True)

	st.markdown("### Correlations and relationships")
	corr_cols = st.multiselect(
		"Select variables",
		options=analytic_cols,
		default=analytic_cols,
	)
	if len(corr_cols) >= 2:
		st.plotly_chart(plot_correlation_heatmap(df_quality, corr_cols), use_container_width=True)
	else:
		st.info("Select at least two numeric variables.")

	st.markdown("### Two-dimensional analysis")
	default_x = analytic_cols.index("alcohol") if "alcohol" in analytic_cols else 0
	default_y = analytic_cols.index("sulphates") if "sulphates" in analytic_cols else 1
	x_feature = st.selectbox("X axis", analytic_cols, index=default_x)
	y_feature = st.selectbox("Y axis", analytic_cols, index=default_y)
	fig_scatter = px.scatter(
		df_quality,
		x=x_feature,
		y=y_feature,
		color="quality_label",
		hover_data=["quality", "alcohol", "sulphates"],
		trendline="ols",
		title="Two-feature relationship",
	)
	st.plotly_chart(fig_scatter, use_container_width=True)

	st.markdown("### Feature distribution across quality labels")
	box_feature_options = [col for col in analytic_cols if col != "quality"]
	box_feature = st.selectbox(
		"Feature for box plot",
		box_feature_options,
		index=box_feature_options.index("alcohol") if "alcohol" in box_feature_options else 0,
	)
	box_fig = px.box(
		df_quality,
		x="quality_label",
		y=box_feature,
		color="quality_label",
		points="outliers",
		title=f"{box_feature} distribution by quality label",
	)
	st.plotly_chart(box_fig, use_container_width=True)

	st.markdown("### Clustering")
	c1, c2 = st.columns(2)
	with c1:
		clusters = st.slider("Number of clusters", min_value=2, max_value=8, value=4)
	with c2:
		random_state = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

	clustered_df, cluster_fig, variance = run_clustering_analysis(df_quality, clusters, random_state)
	st.plotly_chart(cluster_fig, use_container_width=True)

	st.markdown("#### Cluster-level averages")
	st.dataframe(clustered_df.groupby("cluster").mean(numeric_only=True).round(2))
	if variance.size >= 2:
		st.caption(
			f"PCA explained variance ratios: PC1={variance[0]:.2f}, PC2={variance[1]:.2f}"
		)


def render_modeling_page(df_quality: pd.DataFrame) -> None:
	st.header("Predictive modelling")

	st.markdown(
		"Build a binary classifier to separate high-quality wines from the rest."
	)

	col1, col2, col3 = st.columns(3)
	with col1:
		test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
	with col2:
		threshold = st.slider("Quality threshold (1 = high)", 4, 8, 6, 1)
	with col3:
		algorithm = st.selectbox(
			"Algorithm",
			["Logistic Regression", "Random Forest", "XGBoost"],
		)

	use_smote = st.checkbox("Use SMOTE balancing", value=True)
	random_state = st.number_input("Random seed", min_value=0, max_value=9999, value=42)

	df_model = df_quality.copy()
	df_model["target"] = (df_model["quality"] >= threshold).astype(int)

	feature_cols = [c for c in df_model.columns if df_model[c].dtype != "O" and c not in {"target", "quality", "high_quality_flag"}]
	X = df_model[feature_cols]
	y = df_model["target"]

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=test_size, random_state=random_state, stratify=y
	)

	if use_smote:
		smote = SMOTE(random_state=random_state)
		X_train, y_train = smote.fit_resample(X_train, y_train)

	model = build_model(algorithm, int(random_state), num_class=2)

	pipeline = Pipeline([
		("scaler", StandardScaler()),
		("model", model),
	])

	pipeline.fit(X_train, y_train)
	y_pred = pipeline.predict(X_test)
	y_prob = pipeline.predict_proba(X_test)[:, 1]

	accuracy = (y_pred == y_test).mean()
	precision = precision_score(y_test, y_pred, zero_division=0)
	recall = recall_score(y_test, y_pred, zero_division=0)
	f1 = f1_score(y_test, y_pred, zero_division=0)

	metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
	metric_col1.metric("Accuracy", f"{accuracy:.3f}")
	metric_col2.metric("Precision", f"{precision:.3f}")
	metric_col3.metric("Recall", f"{recall:.3f}")
	metric_col4.metric("F1", f"{f1:.3f}")

	st.markdown("#### Stratified five-fold cross-validation")
	cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
	cv_results = cross_val_score(pipeline, X, y, cv=cv, scoring="f1")
	st.write(
		pd.DataFrame(
			{
				"Fold": range(1, len(cv_results) + 1),
				"F1": cv_results,
			}
		).assign(Mean=cv_results.mean(), Std=cv_results.std())
	)

	st.markdown("#### Confusion matrix")
	cm = confusion_matrix(y_test, y_pred)
	st.plotly_chart(plot_confusion_matrix_matrix(cm, labels=["Low", "High"]), use_container_width=False)

	st.markdown("#### ROC curve")
	fpr, tpr, _ = roc_curve(y_test, y_prob)
	roc_auc = auc(fpr, tpr)
	roc_fig = go.Figure()
	roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={roc_auc:.3f})"))
	roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Baseline", line=dict(dash="dash")))
	roc_fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", width=500, height=400)
	st.plotly_chart(roc_fig, use_container_width=False)

	st.markdown("#### Classification report")
	report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
	st.dataframe(pd.DataFrame(report).T.round(3))

	st.markdown("#### Feature relevance (tree-based models)")
	if hasattr(pipeline.named_steps["model"], "feature_importances_"):
		importances = pipeline.named_steps["model"].feature_importances_
		importance_df = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values("importance")
		fig_importance = px.bar(importance_df, x="importance", y="feature", orientation="h", title="Feature importance")
		st.plotly_chart(fig_importance, use_container_width=True)
	elif hasattr(pipeline.named_steps["model"], "coef_"):
		coefs = pipeline.named_steps["model"].coef_[0]
		coef_df = pd.DataFrame({"feature": feature_cols, "coef": coefs}).sort_values("coef")
		fig_coef = px.bar(coef_df, x="coef", y="feature", orientation="h", title="Regression weights")
		st.plotly_chart(fig_coef, use_container_width=True)

	st.markdown("#### Scenario simulator")
	with st.expander("Predict quality for custom measurements", expanded=False):
		# Provide both manual feature entry and sampling of an existing record.
		mode = st.radio("Input mode", ["Manual entry", "Pick existing wine"])
		base_features = [
			"fixed acidity",
			"volatile acidity",
			"citric acid",
			"residual sugar",
			"chlorides",
			"free sulfur dioxide",
			"total sulfur dioxide",
			"density",
			"pH",
			"sulphates",
			"alcohol",
		]
		derived_features = [col for col in feature_cols if col not in base_features]

		if mode == "Pick existing wine":
			row_index = st.slider("Dataset row", 0, len(df_quality) - 1, 0)
			source_row = df_quality.iloc[[row_index]][feature_cols]
		else:
			input_columns = st.columns(3)
			manual_inputs: dict[str, float] = {}
			for idx, column in enumerate(base_features):
				col_widget = input_columns[idx % 3]
				with col_widget:
					default_value = float(df_quality[column].median())
					manual_inputs[column] = st.number_input(
						column,
						value=float(round(default_value, 3)),
						format="%.3f",
					)

			manual_df = pd.DataFrame([manual_inputs])
			fixed_acidity = manual_df["fixed acidity"].replace(0, np.nan)
			total_sulfur = manual_df["total sulfur dioxide"].replace(0, np.nan)
			manual_df["sulphates_to_acidity"] = manual_df["sulphates"] / fixed_acidity
			manual_df["total_sulfur_ratio"] = manual_df["free sulfur dioxide"] / total_sulfur
			manual_df["sugar_density_interaction"] = manual_df["residual sugar"] * manual_df["density"]
			manual_df["alcohol_acidity_ratio"] = manual_df["alcohol"] / fixed_acidity
			for feature in derived_features:
				if feature not in manual_df.columns:
					manual_df[feature] = np.nan
			source_row = manual_df[feature_cols]

		source_row = source_row.fillna(df_quality[feature_cols].median())
		probability = pipeline.predict_proba(source_row)[0][1]
		prediction = pipeline.predict(source_row)[0]
		st.metric("Predicted class", "High quality" if prediction == 1 else "Lower quality")
		st.metric("Probability of high quality", f"{probability:.2%}")
		st.dataframe(source_row)


def render_pairings_page(df_pairings: pd.DataFrame) -> None:
	st.header("Food pairing intelligence")

	col_filters = st.columns(4)
	with col_filters[0]:
		wine_type = st.multiselect("Wine type", sorted(df_pairings["wine_type"].unique()))
	with col_filters[1]:
		wine_category = st.multiselect("Wine category", sorted(df_pairings["wine_category"].unique()))
	with col_filters[2]:
		food_category = st.multiselect("Food category", sorted(df_pairings["food_category"].unique()))
	with col_filters[3]:
		cuisine = st.multiselect("Cuisine", sorted(df_pairings["cuisine"].unique()))

	quality_values = df_pairings["pairing_quality"].dropna()
	if quality_values.empty:
		min_quality, max_quality = 0, 0
	else:
		min_quality, max_quality = int(quality_values.min()), int(quality_values.max())
	selected_quality = st.slider("Pairing quality range", min_quality, max_quality, (min_quality, max_quality), step=1)

	filtered = df_pairings.copy()
	if wine_type:
		filtered = filtered[filtered["wine_type"].isin(wine_type)]
	if wine_category:
		filtered = filtered[filtered["wine_category"].isin(wine_category)]
	if food_category:
		filtered = filtered[filtered["food_category"].isin(food_category)]
	if cuisine:
		filtered = filtered[filtered["cuisine"].isin(cuisine)]
	filtered = filtered[(filtered["pairing_quality"] >= selected_quality[0]) & (filtered["pairing_quality"] <= selected_quality[1])]

	if filtered.empty:
		st.warning("No rows satisfy the current filters.")
		return

	st.markdown("#### Filtered sample")
	st.dataframe(filtered.head(100))
	st.caption(f"Total matches: {len(filtered)}")
	render_download_button(filtered, "Download filtered data", "pairings_filtered.csv")

	st.markdown("### Pairing quality distribution")
	fig_box = px.box(filtered, x="wine_category", y="pairing_quality", color="wine_category", points="all")
	st.plotly_chart(fig_box, use_container_width=True)

	st.markdown("### Top combinations")
	top_pairings = filtered.sort_values("pairing_quality", ascending=False).head(20)
	st.dataframe(top_pairings[["wine_type", "food_item", "pairing_quality", "quality_label", "description"]])

	st.markdown("### Cuisine performance")
	cuisine_summary = (
		filtered.groupby("cuisine")["pairing_quality"].agg(["mean", "count"]).reset_index()
		.rename(columns={"mean": "mean_quality", "count": "pairings"})
		.sort_values("mean_quality", ascending=False)
	)
	bar_cuisine = px.bar(
		cuisine_summary,
		x="mean_quality",
		y="cuisine",
		orientation="h",
		title="Average pairing score by cuisine",
		color="pairings",
		color_continuous_scale="Plasma",
	)
	st.plotly_chart(bar_cuisine, use_container_width=True)
	st.dataframe(cuisine_summary.head(15))

	st.markdown("### Structural treemap")
	treemap_fig = px.treemap(
		filtered,
		path=["wine_type", "food_category", "food_item"],
		values="pairing_quality",
		color="pairing_quality",
		color_continuous_scale="RdYlGn",
		title="Pairing quality structure",
	)
	st.plotly_chart(treemap_fig, use_container_width=True)

	st.markdown("### Frequent words in descriptions")
	top_terms = compute_top_terms(filtered["description"], top_n=20)
	if not top_terms.empty:
		st.plotly_chart(render_word_frequency_bar(top_terms), use_container_width=True)
	else:
		st.info("No descriptions match the current filters.")


def render_cross_insights(df_quality: pd.DataFrame, df_pairings: pd.DataFrame) -> None:
	st.header("Cross-dataset insights")

	st.markdown(
		"""
		The two datasets capture complementary views: laboratory measurements of
		red wines and qualitative feedback on food pairing success. Compare both
		perspectives to generate hypotheses or prioritise product experiments.
		"""
	)

	st.markdown("### Aggregated winequality-red metrics")
	quality_summary = (
		df_quality.groupby("quality_label")
		.agg({"alcohol": "mean", "residual sugar": "mean", "fixed acidity": "mean", "density": "mean"})
		.round(2)
	)
	st.dataframe(quality_summary)

	st.markdown("### Pairing preferences by wine style")
	pair_summary = (
		df_pairings.groupby(["wine_type", "quality_label"]).agg({"pairing_quality": "mean"}).reset_index()
	)
	fig_pair_heatmap = px.density_heatmap(
		pair_summary,
		x="quality_label",
		y="wine_type",
		z="pairing_quality",
		color_continuous_scale="Viridis",
		title="Average pairing quality",
	)
	st.plotly_chart(fig_pair_heatmap, use_container_width=True)

	st.markdown("### Style mapping table")
	mapping = pd.DataFrame(
		{
			"wine_type": df_pairings["wine_type"],
			"wine_category": df_pairings["wine_category"],
			"pairing_quality": df_pairings["pairing_quality"],
		}
	)
	mapping_summary = mapping.groupby(["wine_type", "wine_category"]).pairing_quality.mean().reset_index()
	st.dataframe(mapping_summary.sort_values("pairing_quality", ascending=False).head(20))

	st.markdown("### Insight scratchpad")
	notes = st.text_area(
		"Keep notes locally in the Streamlit session state",
		value=st.session_state.get("notes", ""),
		height=200,
	)
	st.session_state["notes"] = notes


def render_export_page(df_quality: pd.DataFrame, df_pairings: pd.DataFrame) -> None:
	st.header("Data export")

	st.markdown("### Export winequality-red")
	quality_cols = st.multiselect(
		"Columns to export (winequality)",
		options=df_quality.columns.tolist(),
		default=df_quality.columns.tolist(),
	)
	render_download_button(df_quality[quality_cols], "Download winequality-red.csv", "winequality_red_filtered.csv")

	st.markdown("### Export wine_food_pairings")
	pairing_cols = st.multiselect(
		"Columns to export (pairings)",
		options=df_pairings.columns.tolist(),
		default=df_pairings.columns.tolist(),
	)
	render_download_button(df_pairings[pairing_cols], "Download wine_food_pairings.csv", "wine_food_pairings_filtered.csv")


def main() -> None:
	df_quality = load_wine_quality()
	df_pairings = load_pairings()

	page = st.sidebar.radio(
		"Navigation",
		[
			"Overview",
			"Wine Quality",
			"Modelling",
			"Pairings",
			"Cross insights",
			"Export",
		],
	)

	st.sidebar.markdown("---")
	st.sidebar.markdown("### Quick stats")
	st.sidebar.write(f"winequality rows: {len(df_quality)}")
	st.sidebar.write(f"pairings rows: {len(df_pairings)}")
	st.sidebar.write(f"winequality columns: {len(df_quality.columns)}")
	st.sidebar.write(f"pairings columns: {len(df_pairings.columns)}")

	if page == "Overview":
		render_page_intro(df_quality, df_pairings)
	elif page == "Wine Quality":
		render_page_quality(df_quality)
	elif page == "Modelling":
		render_modeling_page(df_quality)
	elif page == "Pairings":
		render_pairings_page(df_pairings)
	elif page == "Cross insights":
		render_cross_insights(df_quality, df_pairings)
	else:
		render_export_page(df_quality, df_pairings)


if __name__ == "__main__":
	main()

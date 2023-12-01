import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
page_title="Data Exploration",
page_icon="📊",
layout="wide",
initial_sidebar_state="expanded")

#The title and text
st.title("Data Exploration 📊 ")
st.write("In this tab we can see the most relevant information that we can extract through the data from the visual analytics.")

def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map


def plot_features_against_target(df):
    target = 'price'
    features = [x for x in df.columns if x not in ["car", "model", "registration", target]]
    n_cols = 3
    n_rows = (len(features) + 2) // n_cols
    sns.set(font_scale=2)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 6))
    axes = axes.flatten()
    
    # Plot each feature against the target variable in the dataframe
    for i, feature in enumerate(features):
        ax = axes[i]
        if df[feature].dtype == 'object' or df[feature].nunique() < 10:
            # For categorical data, use a boxplot or violin plot
            sns.boxplot(x=feature, y=target, data=df, ax=ax)
        else:
            # For numerical data, use a scatter plot
            sns.scatterplot(x=feature, y=target, data=df, ax=ax)
        ax.set_title(f'{feature} vs {target}')
        plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

    # Hide any unused subplots
    for j in range(i + 1, n_rows * n_cols):
        fig.delaxes(axes[j])
    fig.tight_layout()

    return fig

@st.cache_data
def load_data():
    df =  pd.read_csv("car_ad_display.csv", encoding = "ISO-8859-1", sep=";").drop(columns='Unnamed: 0')

    car_map = shorten_categories(df.car.value_counts(), 10)
    df['car'] = df['car'].map(car_map)

    model_map = shorten_categories(df.model.value_counts(), 10)
    df['model'] = df['model'].map(model_map)

    df = df[df["price"] <= 100000]
    df = df[df["price"] >= 1000]
    df = df[df["mileage"] <= 600]
    df = df[df["engV"] <= 7.5]
    df = df[df["year"] >= 1975]

    return df

df = load_data()

#Basic info:
st.subheader("How does the data look like?")
rows = len(df)
columns =len(df.columns)
st.write("We have:")
st.write(str(rows)+" rows")
st.write(str(columns)+" columns.")
st.dataframe(df.head(3))


#Target:
st.subheader("Distribution of the target:")
sns.set(font_scale=0.5)
sns.set_palette("icefire")
fig = plt.figure(figsize=(5, 1.5))
sns.kdeplot(x="price", data=df, fill=True)
plt.title("Prices")
st.pyplot(fig)


#Most expensive cars:
st.subheader("Top 10 Most expensive cars:")
df_priceByCar = df[['car','price']].groupby('car').mean().reset_index()
df_priceByCar = df_priceByCar.sort_values('price', ascending=False).head(10)
fig = plt.figure(figsize=(5, 2))
ax = sns.barplot(df_priceByCar, x="price", y="car", palette= "icefire")
ax.bar_label(ax.containers[0], fontsize=5)
st.pyplot(fig)


#Features distribution vs target:
st.subheader("Distribution of the features against the target:")
fig= plot_features_against_target(df)
st.pyplot(fig)
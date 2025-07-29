import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def preprocess_data(original_dataset, preprocessed_data):
    """
    This function reads the raw data from the input file and 
    performs initial preprocessing steps.
    """
    # Read data from CSV file
    df = pd.read_csv(original_dataset)

    # 3.1 Missing Value Imputation
    # Replace '?' with mode for categorical attributes
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if '?' in df[col].values:
            mode_value = df[col][df[col] != '?'].mode()[0]
            df[col] = df[col].replace('?', mode_value)
            print(f"Imputed missing values in {col} with mode: {mode_value}")

    # 3.2
    # Count occurrences of each country
    country_counts = df['native-country'].value_counts()
    # Identify countries with fewer than 40 occurrences
    infrequent_countries = country_counts[country_counts < 40].index
    # Replace infrequent countries with 'Others'
    df['native-country'] = df['native-country'].replace(infrequent_countries, 'Others')

    # Print remaining unique countries
    remaining_countries = df['native-country'].nunique()
    print(f"\nNumber of unique countries after merging infrequent ones: {remaining_countries}")
    print("Remaining countries:", df['native-country'].unique())

    # 3.3 
    # Create binary columns for each categorical attribute
    binary_df = pd.DataFrame(index=df.index)

    # Process each categorical column
    for col in categorical_cols:
        # Get unique values in the column
        unique_values = df[col].unique()

        # Create binary columns for each value
        for value in unique_values:
            col_name = f"{col}_{value}"
            binary_df[col_name] = (df[col] == value).astype(bool)

    # Add numerical columns as they are
    numerical_cols = df.select_dtypes(include=['int']).columns
    for col in numerical_cols:
        binary_df[col] = df[col]

    # Create output directory if it doesn't exist
    Path(preprocessed_data).parent.mkdir(parents=True, exist_ok=True)

    # Save preprocessed data
    binary_df.to_csv(preprocessed_data, index=False)
    print(f"\nPreprocessed data saved to {preprocessed_data}")

original_dataset = 'data/adult.csv'
preprocessed_data = 'dist/adult_preprocessedd.csv'
preprocess_data(original_dataset, preprocessed_data)

def calculate_split_points(data, attribute, method, n_intervals=None, custom_splits=None):
    """
    This function calculates split points based on the specified method.
    """
    if custom_splits is not None:
        return np.array(custom_splits)
    elif method == 'equal_frequency':
        return np.percentile(data[attribute], np.linspace(0, 100, n_intervals + 1)[1:-1])
    else:  # equal_width
        return np.linspace(data[attribute].min(), data[attribute].max(), n_intervals + 1)[1:-1]


def create_binary_features(data, attribute, split_points):
    """
    This function creates binary features based on intervals.
    """
    binary_features = pd.DataFrame()

    # First interval (-inf, x1]
    col_name = f"{attribute} <= {split_points[0]:.2f}"
    binary_features[col_name] = (data[attribute] <= split_points[0]).astype(bool)

    # Middle intervals (xi, xi+1]
    for i in range(len(split_points) - 1):
        col_name = f"{split_points[i]:.2f} < {attribute} <= {split_points[i + 1]:.2f}"
        binary_features[col_name] = ((data[attribute] > split_points[i]) &
                                     (data[attribute] <= split_points[i + 1])).astype(bool)

    # Last interval (xn-1, inf)
    col_name = f"{attribute} > {split_points[-1]:.2f}"
    binary_features[col_name] = (data[attribute] > split_points[-1]).astype(bool)

    return binary_features


def create_strip_plots(data, attributes_config):
    """
    This function creates strip plots for all specified attributes.
    """
    fig, axes = plt.subplots(len(attributes_config), 1, figsize=(12, 4 * len(attributes_config)))

    for idx, (attr, config) in enumerate(attributes_config.items()):
        ax = axes[idx]

        # Create strip plot
        sns.stripplot(x=data[attr], jitter=True, size=2, alpha=0.7, ax=ax, color='blue')

        # Add split points
        for split_point in config['split_points']:
            ax.axvline(x=split_point, color='green', linestyle='--', alpha=0.7)

        # Add legend 
        legend_labels = [f"{attr}={split_point:.2f}" for split_point in config['split_points']]
        ax.legend(legend_labels, title="Split Points", bbox_to_anchor=(1.05, 1))

        # Set title and labels
        ax.set_title(f'Strip Plot for Attribute "{attr}"')
        ax.set_xlabel(attr)

        # Set log scale if needed
        if config.get('log_scale', False):
            ax.set_xscale('log')

    plt.tight_layout()
    plt.show()


def preprocess_final(original_dataset, preprocessed_data):
    """
    This function reads the preprocessed data from Section 3.3 and
    performs final preprocessing steps.
    """
    # Read preprocessed data from Section 3.3
    df = pd.read_csv(original_dataset)

    # Configuration for attributes
    attributes_config = {
        'age': {'method': 'equal_frequency','n_intervals': 12,'log_scale': False},
        'education-num': {'method': 'equal_width','n_intervals': 8,'log_scale': False},
        'capital-gain': {'method': 'custom','split_points': [2000, 5700, 11500, 21500, 64000],'log_scale': True},
        'capital-loss': {'method': 'custom','split_points': [900, 2000, 3100],'log_scale': True},
        'hours-per-week': {'method': 'equal_width','n_intervals': 5,'log_scale': False}
    }

    # Process each continuous attribute
    binary_features = pd.DataFrame(index=df.index)

    for attr, config in attributes_config.items():
        # Calculate split points based on method
        if config['method'] == 'custom':
            split_points = calculate_split_points(df, attr, config['method'],
                                         custom_splits=config['split_points'])
        else:
            split_points = calculate_split_points(df, attr, config['method'],
                                         n_intervals=config['n_intervals'])

        # Store split points for plotting
        config['split_points'] = split_points

        # Create binary features
        attr_binary = create_binary_features(df, attr, split_points)
        binary_features = pd.concat([binary_features, attr_binary], axis=1)

    # Create visualization
    create_strip_plots(df, attributes_config)

    # Create final dataset
    # Remove original continuous columns and fnlwgt
    continuous_cols = list(attributes_config.keys()) + ['fnlwgt']
    final_df = df.drop(columns=continuous_cols)

    # Add binary features
    final_df = pd.concat([final_df, binary_features], axis=1)

    # Save preprocessed data
    Path(preprocessed_data).parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(preprocessed_data, index=False)

    print(f"Final dataset shape: {final_df.shape}")
    print(f"Number of features: {len(final_df.columns)}")
    print(f"Dataset saved to: {preprocessed_data}")


# Execute preprocessing
original_dataset = 'dist/adult_preprocessedd.csv'  
preprocessed_data = 'dist/adult_preprocessed.csv'
preprocess_final(original_dataset, preprocessed_data)
import numpy as np
import pandas as pd
import pingouin as pg
import pymannkendall as mk
import scikit_posthocs as sp
from scipy.stats import *
from sklearn.preprocessing import PowerTransformer
import statsmodels.formula.api as smf
import statsmodels.api as sm
import statsmodels.formula.api as ols


global alpha # Significance level for hypothesis tests
alpha = 0.05


def fligner_test(a, b):
    """
    Perform the Fligner-Killeen test to compare the variances of two samples.
    Parameters:
        a (array-like): First sample data.
        b (array-like): Second sample data.
    Returns:
        None
            Displays the Fligner-Killeen statistic and p-value, and interprets the result.
    """
    # Run the Fligner-Killeen test
    statistic, p_value = fligner(a, b)

    # Display the results
    print(f"Fligner-Killeen Statistic: {statistic}")
    print(f"P-value: {p_value}")

    # Interpret the result
    if p_value < alpha:
        print("The variances are statistically significantly different (reject null hypothesis).")
        print("Run Kolmogorov-Smirnov test.")
    elif p_value < 0.1:
        print("The variances may be different (weak evidence against the null hypothesis).")
        print("Run Kolmogorov-Smirnov test.")
    else:
        print("The variances are not statistically significantly different (fail to reject null hypothesis).")
        print("Run Mann-Whitney U test.")


def check_normality(*arrays, verbose=True):
    """
    Check the normality and lognormality of multiple arrays using Shapiro-Wilk or D'Agostino & Pearson tests.
    Parameters:
        *arrays : array-like
            One or more arrays to be tested for normality.
        verbose : bool
            If True, prints the test results and interpretation for each array.
    Returns:
        list
            A list of booleans indicating whether each array is normal (True) or not (False).
    """
    normality_results = []

    # Replace None with np.nan
    arrays = [np.array(arr) for arr in arrays]
    arrays = [np.array(arr, dtype=float) for arr in arrays]
    arrays = [np.where(arr == None, np.nan, arr) for arr in arrays]

    # Flatten each array and remove NaN values
    flattened_arrays = [arr.flatten() for arr in arrays]
    flattened_arrays = [arr[~np.isnan(arr)] for arr in flattened_arrays]
    
    # Compare the lengths of these arrays
    for i, values in enumerate(flattened_arrays):
        # Run the appropriate normality test based on the longest length
        if len(values) < 50:
            stat, p = shapiro(values)
            test_name = "Shapiro-Wilk"
        else:
            stat, p = normaltest(values)
            test_name = "D'Agostino & Pearson"
        
        if verbose:
            print(f'{test_name} test for array {i+1} - Statistics={stat:.5f}, p={p:.5f}')
        
        # Interpret the result
        is_normal = p > alpha
        normality_results.append(is_normal)
        
        if verbose:
            if is_normal:
                print(f'Sample for array {i+1} looks Gaussian (normal)\n')
            else:
                print(f'Sample for array {i+1} does not look Gaussian (not normal)\n')

        # Check for lognormality
        if np.any(values <= 0):
            if verbose:
                print(f'Skipping lognormality test for array {i+1} due to non-positive values\n')
            continue
        else:
            log_values = np.log(values)  # Log-transform the values
            if len(log_values) < 50:
                stat, p = shapiro(log_values)
                test_name = "Shapiro-Wilk"
            else:
                stat, p = normaltest(log_values)
                test_name = "D'Agostino & Pearson"
            
            if verbose:
                print(f'{test_name} test for log-transformed array {i+1} - Statistics={stat:.5f}, p={p:.5f}')
            
            # Interpret the result
            if verbose:
                if p > alpha:
                    print(f'Sample for log-transformed array {i+1} looks Gaussian (lognormal)\n')
                else:
                    print(f'Sample for log-transformed array {i+1} does not look Gaussian (not lognormal)\n')

    return normality_results


def spearman_correlation(a, b, verbose=True):
    """
    Calculate the Spearman correlation coefficient between two arrays.
    Parameters:
        a (array-like): First array.
        b (array-like): Second array.
    Returns:
        tuple
            Spearman correlation coefficient and p-value.
    """
    correlation, p_value = spearmanr(a, b, nan_policy='omit')

    if verbose:
        print(f"Spearman correlation: {correlation:.5f}")
        print(f"P-value: {p_value:.5f}")
    
        if p_value < alpha:
            print("The correlation is statistically significant.")
        else:
            print("The correlation is not statistically significant.")
    
    return correlation, p_value


def spearman_correlation_repeated_measures(df, verbose=True, return_results=True):
    """
    Calculate the Spearman correlation coefficient for repeated measures data.
    Parameters:
        df (DataFrame): DataFrame where each column represents a different subject and each row represents a repeated measure.
        verbose (bool): If True, prints the correlation and p-value for each measure.
        return_results (bool): If True, returns the correlations and p-values as lists.
    Returns:
        tuple (optional)
            If return_results is True, returns a tuple containing the list of Spearman correlation coefficients and p-values across all measures.
    """
    correlations = list()
    p_values = list()

    for col in df.columns:
        if df[col].notnull().any():
            correlation, p_value = spearman_correlation(list(df[col].index.values), list(df[col]), verbose=False)
            correlations.append(correlation)
            p_values.append(p_value)

    # Remove NaNs before calculating median
    correlations = [corr for corr in correlations if not np.isnan(corr)]
    p_values = [p for p in p_values if not np.isnan(p)]

    if verbose:
        print(f"Median correlation: {np.median(correlations):.5f}")
        print(f"Median p-value: {np.median(p_values):.5f}")

        if np.median(p_values) < alpha:
            print("The median correlation is statistically significant.")
        else:
            print("The median correlation is not statistically significant.")

    if return_results:
        return correlations, p_values


def pearson_correlation(a, b, verbose=True):
    """
    Calculate the Pearson correlation coefficient between two arrays.
    Parameters:
        a (array-like): First array.
        b (array-like): Second array.
    Returns:
        tuple
            Pearson correlation coefficient and p-value.
    """
    correlation, p_value = pearsonr(a, b)

    if verbose:
        print(f"Pearson correlation: {correlation:.5f}")
        print(f"P-value: {p_value:.5f}")
    
        if p_value < alpha:
            print("The correlation is statistically significant.")
        else:
            print("The correlation is not statistically significant.")
    
    return correlation, p_value


def pearson_correlation_repeated_measures(df):
    """
    Calculate the Pearson correlation coefficient for repeated measures data.
    Parameters:
        df (DataFrame): DataFrame where each column represents a different subject and each row represents a repeated measure.
    Returns:
        None
            Prints the mean Pearson correlation coefficient and mean p-value across all measures.
    """
    correlations = list()
    p_values = list()

    for col in df.columns:
        if df[col].notnull().any():
            correlation, p_value = pearson_correlation(list(df[col].index.values), list(df[col]), verbose=False)
            correlations.append(correlation)
            p_values.append(p_value)

    # Remove NaNs before calculating median
    correlations = [corr for corr in correlations if not np.isnan(corr)]
    p_values = [p for p in p_values if not np.isnan(p)]

    print(f"Median correlation: {np.median(correlations):.5f}")
    print(f"Median p-value: {np.median(p_values):.5f}")

    if np.mean(p_values) < alpha:
        print("The median correlation is statistically significant.")
    else:
        print("The median correlation is not statistically significant.")

    return correlations, p_values


def hamed_rao_modified_mann_kendall_test(df, verbose=True, return_results=False):
    """
    Run the Hamed and Rao Modified Mann-Kendall test for non-normal, repeated measures data.
    Parameters:
        df (DataFrame): DataFrame where each column represents a different subject and each row represents a repeated measure.
        verbose (bool): If True, prints the tau and p-value for each measure.
        return_results (bool): If True, returns the taus and p-values as lists.
    Returns:
        tuple (optional)
            If return_results is True, returns a tuple containing the list of taus and p-values across all measures.
    """
    taus = list()
    p_values = list()

    for col in df.columns:
        if df[col].notnull().any():
            result = mk.hamed_rao_modification_test(list(df[col]))
            taus.append(result.Tau)
            p_values.append(result.p)

    # Remove NaNs before calculating median
    taus = [corr for corr in taus if not np.isnan(corr)]
    p_values = [p for p in p_values if not np.isnan(p)]

    if verbose:
        print(f"Median tau: {np.median(taus):.5f}")
        print(f"Median p-value: {np.median(p_values):.5f}")

        if np.median(p_values) < alpha:
            print("The median trend is statistically significant.")
        else:
            print("The median trend is not statistically significant.")

    if return_results:
        return taus, p_values


def check_monotonicity_repeated_measures(df, return_results=False):
    """
    Check the monotonicity of a repeated measures DataFrame.
    Parameters:
        df (DataFrame): DataFrame where each column represents a different subject and each row represents a repeated measure.
        return_results (bool): If True, returns the correlations and p-values.
    Returns:
        tuple (optional)
            If return_results is True, returns a tuple containing the correlations and p-values.
    """
    normality = check_normality(df, verbose=False)
    
    # Interpret the result and run the appropriate correlation test
    if normality[0]:
        print('Data looks Gaussian (normal). Running Pearson correlation.')
        correlations, p_values = pearson_correlation_repeated_measures(df)
    else:
        print('Data does not look Gaussian (not normal). Running Spearman correlation.')
        correlations, p_values = spearman_correlation_repeated_measures(df)

    if return_results:
        return correlations, p_values
    

def boxcox_transformation(data):
    """
    Apply the Box-Cox transformation to the data.
    Parameters:
        data (array-like): Data to be transformed.
    Returns:
        array-like
            Transformed data.
    """
    return boxcox(data)[0]


def yeo_johnson_transformation(data):
    """
    Apply the Yeo-Johnson transformation to the data.
    Parameters:
        data (array-like): Data to be transformed.
    Returns:
        array-like
            Transformed data.
    """
    transformer = PowerTransformer(method='yeo-johnson')
    return transformer.fit_transform(data.reshape(-1, 1))


def transform_to_parametric(*data, force_boxcox=False, shift=0.0001):
    """
    Apply the Box-Cox or Yeo-Johnson transformation to the data to make it more Gaussian-like.
    Parameters:
        *data : array-like
            One or more arrays to be transformed.
        force_boxcox : bool, optional
            If True, forces the use of Box-Cox transformation even if data contains non-positive values.
        shift : float, optional
            The value to shift the data by to make all values positive for Box-Cox transformation.
    Returns:
        list
            A list of transformed arrays.
    """
    transformed_data = []

    for arr in data:   
        arr = np.array(arr)
        if np.any(arr <= 0):
            if force_boxcox:
                print("Running Box-Cox transformation with shift")
                arr = arr + np.abs(np.min(arr)) + shift
                transformed_data.append(boxcox_transformation(arr))
            else:
                print("Running Yeo-Johnson transformation")
                transformed_data.append(yeo_johnson_transformation(arr))
        else:
            print("Running Box-Cox transformation")
            transformed_data.append(boxcox_transformation(arr))
        
    return transformed_data


def kolmogorov_smirnov_test(a, b, verbose=True):
    """
    Perform the Kolmogorov-Smirnov test to compare two samples.
    Parameters:
        a (array-like): First sample data.
        b (array-like): Second sample data.
    Returns:
        tuple
            Kolmogorov-Smirnov statistic and p-value.
    """
    statistic, p_value = ks_2samp(a, b)

    if verbose:
        print(f"Kolmogorov-Smirnov Statistic: {statistic:.5f}")
        print(f"P-value: {p_value:.5f}")
    
        if p_value < alpha:
            print("The distributions are statistically significantly different.")
        else:
            print("The distributions are not statistically significantly different.")
    
    return statistic, p_value


def cohens_d(a, b, use_hedges=False):
    """
    Compute Cohen's d (and optionally Hedges' g) effect size for two independent samples.
    Parameters:
        a (array-like): First sample data.
        b (array-like): Second sample data.
        use_hedges (bool): If True, apply Hedges' g correction for small sample bias.
    Returns:
        tuple: (Effect size, interpretation)
    """
    a, b = np.array(a), np.array(b)  # Convert to NumPy arrays
    mean_diff = np.mean(a) - np.mean(b)
    
    # Compute pooled standard deviation (unbiased estimator)
    n_a, n_b = len(a), len(b)
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

    d = mean_diff / pooled_std

    # Apply Hedges' correction for small sample sizes
    if use_hedges:
        correction_factor = 1 - (3 / (4 * (n_a + n_b) - 9))
        d *= correction_factor
        method = "Hedges' g"
    else:
        method = "Cohen's d"

    print(f"{method}: {d:.5f}")

    # Interpretation based on absolute value
    abs_d = abs(d)
    if abs_d < 0.2:
        print("The effect size is negligible.")
    elif abs_d < 0.5:
        print("The effect size is small.")
    elif abs_d < 0.8:
        print("The effect size is medium.")
    else:
        print("The effect size is large.")

    return d


def cliffs_delta(a, b):
    """
    Compute Cliff's Delta effect size for two independent samples.
    Parameters:
        a (array-like): First sample data.
        b (array-like): Second sample data.
    Returns:
        tuple: (Cliff's Delta, effect size interpretation)
    """
    a, b = np.array(a), np.array(b)  # Convert to numpy arrays
    n_x, n_y = len(a), len(b)

    # Compute rank comparisons using broadcasting
    rank_sum = np.sum(a[:, None] > b) - np.sum(a[:, None] < b)
    delta = rank_sum / (n_x * n_y)
    print(f"Cliff's Delta: {delta:.5f}")

    # Interpretation of effect size (use absolute value for categorization)
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        print("The effect size is negligible.")
    elif abs_delta < 0.33:
        print("The effect size is small.")
    elif abs_delta < 0.474:
        print("The effect size is medium.")
    else:
        print("The effect size is large.")

    return delta


def eta_squared(df, dv, ivs):
    """
    Compute Eta-Squared effect size for ANOVA.
    Parameters:
        df (DataFrame): Pandas DataFrame containing data.
        dv (str): Dependent variable (response variable).
        ivs (list of str): Independent variable(s) (factors).
    Returns:
        dict: Eta-squared values for each independent variable.
    """
    formula = f"{dv} ~ {' + '.join(['C(' + iv + ')' for iv in ivs])}"
    model = ols.ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)  # Type 2 ANOVA (more robust for unbalanced data)

    ss_total = sum(anova_table['sum_sq'])  # Total Sum of Squares
    eta_sq_values = {factor: anova_table.loc[factor, 'sum_sq'] / ss_total for factor in anova_table.index[:-1]}  # Exclude residual

    # Print results
    for factor, eta_sq in eta_sq_values.items():
        print(f"Eta-Squared for {factor}: {eta_sq:.5f}")

        # Interpretation of effect size
        if eta_sq < 0.01:
            print(f"  → Effect size for {factor} is negligible.")
        elif eta_sq < 0.06:
            print(f"  → Effect size for {factor} is small.")
        elif eta_sq < 0.14:
            print(f"  → Effect size for {factor} is medium.")
        else:
            print(f"  → Effect size for {factor} is large.")

    return eta_sq_values


def epsilon_squared(*arrays):
    """
    Compute Epsilon-Squared effect size for a Kruskal-Wallis test.
    Parameters:
        *groups: list-like, each group contains numerical values.
    Returns:
        float: Epsilon-Squared effect size.
    """
    k = len(arrays)  # Number of groups
    n = sum(len(g) for g in arrays)  # Total sample size
    H, p = kruskal(*arrays)  # Compute Kruskal-Wallis test statistic

    # Compute Epsilon-Squared effect size
    epsilon_sq = (H - k + 1) / (n - k)

    print(f"Kruskal-Wallis H: {H:.3f}, p-value: {p:.5f}")
    print(f"Epsilon-Squared (ε²): {epsilon_sq:.5f}")

    # Interpret effect size
    if epsilon_sq < 0.01:
        print("The effect size is negligible.")
    elif epsilon_sq < 0.06:
        print("The effect size is small.")
    elif epsilon_sq < 0.14:
        print("The effect size is medium.")
    else:
        print("The effect size is large")

    return epsilon_sq


def kendalls_w(*arrays):
    """
    Compute Kendall's W for a Friedman test (Repeated Measures ANOVA).
    Parameters:
        *arrays: lists or arrays representing different conditions in repeated measures.
    Returns:
        float: Kendall's W effect size.
    """
    k = len(arrays)  # Number of conditions
    N = len(arrays[0])  # Number of subjects (assuming equal sample sizes across conditions)

    # Compute Friedman test
    friedman_stat, p_value = friedmanchisquare(*arrays)

    # Calculate Kendall's W
    W = friedman_stat / (N * (k - 1))
    
    print(f"Friedman Test Chi-Square: {friedman_stat:.3f}, p-value: {p_value:.3f}")
    print(f"Kendall’s W: {W:.3f}")

    # Interpret effect size
    if W < 0.1:
        print("The effect size is negligible.")
    elif W < 0.3:
        print("The effect size is small.")
    elif W < 0.5:
        print("The effect size is medium.")
    else:
        print("The effect size is large")

    return W


def repeated_measures_anova(df):
    """
    Perform repeated-measures ANOVA on a dataframe where:
    - Each row represents a time point (repeated measure).
    - Each column represents an individual sample (subject).
    Parameters:
        df (pd.DataFrame): DataFrame where rows are repeated measures (time points)
                           and columns are individual samples (subjects).
    Returns:
        pd.DataFrame: ANOVA results including F-statistic, p-value, and effect size.
    """
    # Convert wide format to long format
    df_long = df.reset_index().melt(id_vars=["index"], var_name="Subject", value_name="Value")
    df_long.rename(columns={"index": "Time"}, inplace=True)  # Rename index to Time
    df_long["Value"] = pd.to_numeric(df_long["Value"])

    # Run repeated-measures ANOVA
    anova_results = pg.rm_anova(dv="Value", within="Time", subject="Subject", data=df_long, detailed=True, correction=True)
    corrected_p_value = anova_results.loc[0, "p-GG-corr"]
    print(f"Repeated measures ANOVA p-value: {corrected_p_value}")

    if corrected_p_value < alpha:
        print("The repeated measures ANOVA is statistically significant.")
    else:
        print("The repeated measures ANOVA is not statistically significant.")
    
    return anova_results, corrected_p_value


def likelihood_ratio_test(full_model, reduced_model):
    """
    Perform a likelihood ratio test to compare two nested statistical models.
    The likelihood ratio test evaluates whether the additional parameters 
    in the full model significantly improve the model fit compared to the 
    reduced model. It calculates the likelihood ratio statistic, degrees 
    of freedom difference, and the corresponding p-value.
    Parameters:
        full_model (statsmodels.base.model.Results): The fitted full model 
            (with all parameters included).
        reduced_model (statsmodels.base.model.Results): The fitted reduced 
            model (a nested model with fewer parameters).
    Returns:
        float: The p-value of the likelihood ratio test, indicating the 
        statistical significance of the difference between the two models.
    """
    lr_stat = 2 * (full_model.llf - reduced_model.llf)
    df_diff = full_model.df_modelwc - reduced_model.df_modelwc
    p_value = chi2.sf(lr_stat, df_diff)

    print(f"Likelihood Ratio Statistic: {lr_stat:.3f}")
    print(f"Degrees of Freedom: {df_diff}")
    print(f"P-value: {p_value:.5f}")
    
    return p_value


def mixed_effects_model(df, ind_var_1, ind_var_2, subject, ind_var_1_reference=None, ind_var_2_reference=None):
    """
    Perform mixed-effects model on a dataframe where:
    - Each row represents a time point (repeated measure).
    - Each column represents an individual sample (subject).
    Parameters:
        df (pd.DataFrame): DataFrame where rows are repeated measures (time points)
                           and columns are individual samples (subjects).
        ind_var_1 (str): Name of the first independent variable.
        ind_var_2 (str): Name of the second independent variable.
        subject (str): Name of the subject variable.
        ind_var_1_reference (str, optional): Reference level for the first independent variable.
        ind_var_2_reference (str, optional): Reference level for the second independent variable.
    Returns:
        statsmodels.regression.mixed_linear_model.MixedLMResults: Fitted mixed-effects model results.
    """
    df_long = df.reset_index().melt(id_vars="index", var_name=ind_var_1, value_name="Value")
    df_long = df_long[df_long["Value"].notna()]
    df_long = df_long.reset_index(drop=True)
    df_long[subject] = (df_long.index // 5) + 1
    df_long.rename(columns={df_long.columns[0]: ind_var_2}, inplace=True) 
    df_long["Value"] = pd.to_numeric(df_long["Value"])

    if ind_var_1_reference is None:
        ind_var_1_effect = f"C({ind_var_1})"
    else:
        ind_var_1_effect = f"C({ind_var_1}, Treatment(reference='{ind_var_1_reference}'))"

    if ind_var_2_reference is None:
        ind_var_2_effect = f"C({ind_var_2})"
    else:
        ind_var_2_effect = f"C({ind_var_2}, Treatment(reference={ind_var_2_reference}))"

    # Fit full model with interaction
    model_full = smf.mixedlm(f"Value ~ {ind_var_1_effect} * {ind_var_2_effect}", 
                        data=df_long, groups=subject)
    result_full = model_full.fit(reml=True)

    # Fit reduced model without interaction
    model_reduced = smf.mixedlm(f"Value ~ {ind_var_1_effect} + {ind_var_2_effect}", 
                            data=df_long, groups=subject)
    result_reduced = model_reduced.fit(reml=True)

    if likelihood_ratio_test(result_full, result_reduced) < alpha:
        print("The interaction effect is statistically significant.")
        return result_full
    else:
        # Fit reduced model without independent variable 1
        model_var1_reduced = smf.mixedlm(f"Value ~ {ind_var_2_effect}", data=df_long, groups=subject)
        result_var1_reduced = model_var1_reduced.fit(reml=False)

        # Fit reduced model without independent variable 2
        model_var2_reduced = smf.mixedlm(f"Value ~ {ind_var_1_effect}", data=df_long, groups=subject)
        result_var2_reduced = model_var2_reduced.fit(reml=False)
    
        var1_effect = False
        var2_effect = False
        print(f"Testing Main Effect of {ind_var_1}:")
        if likelihood_ratio_test(result_reduced, result_var1_reduced) < alpha:
            print(f"The main effect of {ind_var_1} is statistically significant.")
            var1_effect = True
        
        print(f"Testing Main Effect of {ind_var_2}:")
        if likelihood_ratio_test(result_reduced, result_var2_reduced) < alpha:
            print(f"The main effect of {ind_var_2} is statistically significant.")
            var2_effect = True
            
        if var1_effect or var2_effect:
            return result_reduced
        else:
            print("No main effects were found.")


def friedman_test(df):
    """
    Perform Friedman test for non-parametric repeated measures analysis.
    Parameters:
        df (pd.DataFrame): Rows are repeated measures (time points), columns are individual samples (subjects).
    Returns:
        dict: Dictionary with Friedman test statistic, p-value, and interpretation.
    """
    # Convert dataframe to list of lists (each row is a time point, each column is a subject)
    values = df.T.values  # Transpose so each subject is a row

    # Run Friedman test
    stat, p_value = friedmanchisquare(*values)

    print(f"Friedman test p-value: {p_value}")

    if p_value < alpha:
        print("The Friedman test is statistically significant.")
    else:
        print("The Friedman test is not statistically significant.")
    
    return stat, p_value


def oneway_anova_repeated_measures(df, return_results=False, multiple_comparisons=True, **kwargs):
    """
    Run one-way ANOVA on a repeated measures DataFrame.
    Parameters:
        df (DataFrame): DataFrame where each column represents a different subject and each row represents a repeated measure.
        return_results (bool): If True, returns the ANOVA statistics and p-values.
        multiple_comparisons (bool): If True, performs post-hoc tests to compare all pairs of time points.
        **kwargs: Additional keyword arguments to pass to the multiple comparisons test.
    Returns:
        tuple (optional)
            If return_results is True, returns a tuple containing the test statistics, p-values, and optionally multiple comparisons results.
    """
    normality = check_normality(df, verbose=False)
    
    # Run the appropriate one-way ANOVA test
    if normality[0]:
        print('Data looks Gaussian (normal). Running repeated-measures ANOVA.')
        stats, p_value = repeated_measures_anova(df)
        if multiple_comparisons and p_value < alpha:
            print("Running Šídák multiple comparisons test")
            mc_results = sidak_multiple_comparisons_test(df, parametric=True, **kwargs)
    
    else:
        print('Data does not look Gaussian (not normal). Running the Friedman test.')
        stats, p_value = friedman_test(df)
        if multiple_comparisons and p_value < alpha:
            print("Running Dunn's multiple comparisons test")
            mc_results = dunns_multiple_comparisons_test(df, **kwargs)

    if return_results:
        if multiple_comparisons and p_value < alpha:
            return stats, p_value, mc_results
        else:
            return stats, p_value
    

def dunns_multiple_comparisons_test(df, control=None):
	"""
	Perform Dunn's test for multiple comparisons.
	Parameters:
		df (pd.DataFrame): DataFrame with rows as repeated measures (time points) 
							and columns as individual samples (subjects).
		control (optional): Specify a control time point (must match one of the DataFrame's index values).
							If provided, only comparisons between the control and all other time points are returned.
							If None, all pairwise comparisons between time points are returned.
	Returns:
		pd.DataFrame or pd.Series:
			- If control is None, returns a DataFrame with Dunn's test p-values for all pairwise comparisons.
			- If a control is specified, returns a Series with p-values comparing each time point to the control.
	"""
	# Convert wide format (time points x subjects) to long format
	df_long = df.reset_index().melt(id_vars='index', var_name='Subject', value_name='Value')
	df_long.rename(columns={'index': 'Time'}, inplace=True)
	df_long["Value"] = pd.to_numeric(df_long["Value"])

	# Run Dunn's test (scikit_posthocs expects long-form data with a grouping variable)
	dunn_results = sp.posthoc_dunn(df_long, val_col='Value', group_col='Time')
	adjusted_dunn_results = dunn_results * len(dunn_results)
	adjusted_dunn_results = adjusted_dunn_results.applymap(lambda x: 0.9999 if x > 1 else x)

	if control is not None:
		if control not in adjusted_dunn_results.index:
			raise ValueError("Specified control time point not found in the DataFrame's index.")
		# Return comparisons of the control against every other time point (exclude self-comparison)
		control_comparisons = adjusted_dunn_results.loc[control].drop(control)
		return control_comparisons
	else:
		# Return full pairwise comparison matrix
		return adjusted_dunn_results


def sidak_multiple_comparisons_test(df, factor_name="Factor", factor_reference=None, time_reference=None, interaction=False, **kwargs):
    """
    Perform Šídák test for multiple comparisons.
    Parameters:
        df (pd.DataFrame): DataFrame with rows as repeated measures (time points) 
                            and columns as individual samples (subjects).
        control (optional): Specify a control time point (must match one of the DataFrame's index values).
                            If provided, only comparisons between the control and all other time points are returned.
                            If None, all pairwise comparisons between time points are returned.
    Returns:
        pd.DataFrame or pd.Series:
            - If control is None, returns a DataFrame with Dunn's test p-values for all pairwise comparisons.
            - If a control is specified, returns a Series with p-values comparing each time point to the control.
    """
    # Convert wide format (time points x subjects) to long format
    df_long = df.reset_index().melt(id_vars="index", var_name=factor_name, value_name="Value")
    df_long = df_long[df_long["Value"].notna()]
    df_long = df_long.reset_index(drop=True)
    df_long["Subject"] = (df_long.index // 5) + 1
    df_long.rename(columns={df_long.columns[0]: "Time"}, inplace=True) 
    df_long["Value"] = pd.to_numeric(df_long["Value"])

    # Run pairwise t-tests for repeated measures using Šídák correction
    sidak_results = pg.pairwise_ttests(dv='Value', within='Time', subject='Subject', 
                                       between=factor_name, data=df_long, padjust='sidak', 
                                       interaction=interaction, **kwargs)

    if interaction:
        # Show only the interaction terms
        sidak_results = sidak_results[sidak_results["Contrast"] == f"Time * {factor_name}"]
        if factor_reference is not None:
            sidak_results = sidak_results[(sidak_results["A"] == factor_reference) | (sidak_results["B"] == factor_reference)]

        sidak_results_rev = pg.pairwise_ttests(dv='Value', within='Time', subject='Subject', 
                                       between=factor_name, data=df_long, padjust='sidak', 
                                       interaction=interaction, within_first=False, **kwargs)
        sidak_results_rev = sidak_results_rev[sidak_results_rev["Contrast"] == f"{factor_name} * Time"]
        if time_reference is not None:
            sidak_results_rev = sidak_results_rev[(sidak_results_rev["A"] == time_reference) | (sidak_results_rev["B"] == time_reference)]

        return pd.concat([sidak_results, sidak_results_rev], ignore_index=True)
        
    else:
        results = []
        if time_reference is not None:
            results.append(sidak_results[(sidak_results["Contrast"] == "Time") & ((sidak_results["A"] == time_reference) | (sidak_results["B"] == time_reference))])
        else:
            results.append(sidak_results[sidak_results["Contrast"] == "Time"])

        if factor_reference is not None:
            results.append(sidak_results[(sidak_results["Contrast"] == factor_name) & ((sidak_results["A"] == factor_reference) | (sidak_results["B"] == factor_reference))])
        else:
            results.append(sidak_results[sidak_results["Contrast"] == factor_name])

        return pd.concat(results, ignore_index=True)
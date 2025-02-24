library(glmnet)
library(randomForest)

#' Preprocess Null Values in Data
#'
#' This function preprocesses a dataset by handling null (NA) values. It replaces missing values in numeric columns with the median of the column, missing values in factor columns with the mode (most common value), and missing values in character columns with the mode.
#'
#' @param data A data frame containing the dataset with potential null values to be preprocessed.
#' @return A list containing the preprocessed predictors.
#' @details This function replaces missing values in numeric columns with the median of the column,
#'          in factor columns with the mode (most common value), and in character columns with the mode as well.
#' @examples
#' # Example 1: Numeric Data with Missing Values
#' set.seed(123)
#' numeric_data <- data.frame(A = c(1, 2, NA, 4, 5), B = c(NA, 2, 3, 4, 5))
#' result1 <- Preprocessing_nullvalues(numeric_data)
#' print(result1)
#'
#' @examples
#' # Example 2: Factor Data with Missing Values
#' factor_data <- data.frame(A = c("A", "B", NA, "A", "B"), B = c("B", "A", "A", "B", NA))
#' result2 <- Preprocessing_nullvalues(factor_data)
#' print(result2)
#'
#' @examples
#' # Example 3: Character Data with Missing Values
#' character_data <- data.frame(A = c("Apple", "Banana", NA, "Apple", "Banana"), B = c("Banana", "Apple", "Apple", "Banana", NA))
#' result3 <- Preprocessing_nullvalues(character_data)
#' print(result3)
#'
#' @examples
#' # Example 4: Mixed Data Types with Missing Values
#' mixed_data <- data.frame(A = c(1, "B", NA, "A", 5), B = c("B", 2, 3, NA, "E"))
#' result4 <- Preprocessing_nullvalues(mixed_data)
#' print(result4)
#'
#' @export
Preprocessing_nullvalues = function(data){
  # Calculate the percentage of NA in each column
  #na_percentage <- colSums(is.na(data)) / nrow(data)
  #removed_columns <- names(data)[na_percentage > 0.6]
  # Remove columns where the percentage of NA is greater than the threshold
  #data <- data[, na_percentage <= 0.6]
  # Impute missing values for remaining columns
  for (col_name in names(data)) {
    if (is.numeric(data[[col_name]])) {
      # Impute numeric columns with the median of the column
      data[[col_name]][is.na(data[[col_name]])] <- median(data[[col_name]], na.rm = TRUE)
    } else if (is.factor(data[[col_name]])) {
      # Impute factor columns with the mode (most common value)
      mode_value <- names(which.max(table(data[[col_name]], useNA = "no")))
      data[[col_name]][is.na(data[[col_name]])] <- mode_value
      data[[col_name]] <- factor(data[[col_name]])  # Re-factor to include new levels if necessary
    } else if (is.character(data[[col_name]])) {
      # Impute character columns with the mode (most common value)
      mode_value <- names(which.max(table(data[[col_name]], useNA = "no")))
      data[[col_name]][is.na(data[[col_name]])] <- mode_value
    }
  }
  return(list(predictors = data))
}

#' One-Hot Encode Categorical Variables
#'
#' This function performs one-hot encoding on categorical variables in a dataset. It converts factor or character columns into dummy variables, creating binary columns for each level of the categorical variable.
#'
#' @param data A data frame containing the dataset to be encoded.
#' @param ref_data (Optional) A reference data frame used to define levels for categorical variables.
#' @param max_levels The maximum number of levels to consider for automatic conversion of character columns to factors(dummy variables).
#' @return returns a data frame containing the original numeric columns along with the one-hot encoded factor columns.
#' @details This function separates the input data frame into numeric and factor/character columns based on
#'          provided reference data or maximum levels criteria. It then converts character columns to factors using the
#'          correct levels from the reference data (if provided) or based on the maximum number of levels criteria and
#'          then performs one-hot encoding on the factor variables.The resulting data frame includes both the original
#'          numeric columns and the one-hot encoded factor columns.
#' @examples
#' # Example 1: Encoding Factor Data
#' factor_data <- data.frame(A = factor(c("red", "blue", "green", "red", "green")),B = factor(c("small", "medium", "large", "medium", "small")))
#' encoded_data1 <- one_hot_encode(factor_data)
#' print(encoded_data1)
#'
#' @examples
#' # Example 2: Encoding Character Data
#' character_data <- data.frame(A = c("apple", "banana", "apple", "banana", "apple"),B = c("small", "medium", "large", "medium", "small"))
#' encoded_data2 <- one_hot_encode(character_data)
#' print(encoded_data2)
#'
#' @examples
#' # Example 3: Encoding Factor Data with Reference Data
#' ref_data <- data.frame(A = factor(c("red", "blue", "green")), B = factor(c("small", "medium", "large")))
#' encoded_data3 <- one_hot_encode(factor_data, ref_data = ref_data)
#' print(encoded_data3)
#'
#' @examples
#' # Example 4: Encoding Character Data with Max Levels
#' encoded_data4 <- one_hot_encode(character_data, max_levels = 3)
#' print(encoded_data4)
#'
#' @examples
#' # Example 5: Numeric Data Only (No Encoding)
#' numeric_data <- data.frame(C = c(1, 2, 3, 4, 5), D = c(6, 7, 8, 9, 10))
#' encoded_data5 <- one_hot_encode(numeric_data)
#' print(encoded_data5)
#'
#' @export
one_hot_encode <- function(data, ref_data=NULL, max_levels=5) {
  if (!is.null(ref_data)) {
    ref_data = as.data.frame(ref_data)
    # Define levels from the reference data
    levels_list <- lapply(ref_data, function(x) {
      if (is.factor(x) || is.character(x)) levels(factor(x))
      else NULL
    })
  }

  data = as.data.frame(data)
  #print(str(data))
  # Separate numeric and factor/character columns
  numeric_data <- data[sapply(data, is.numeric)]
  factor_data <- data[sapply(data, function(x) is.factor(x) || is.character(x))]

  # Convert character columns to factors using the correct levels or based on max_levels criteria
  #prev_names <- names(factor_data)
  processed_data <- lapply(names(factor_data), function(col_name) {
    column <- factor_data[[col_name]]
    if (!is.null(ref_data) && !is.null(levels_list[[col_name]])) {
      # Set factor levels based on the training data
      column <- factor(column, levels = levels_list[[col_name]])
    } else if (length(unique(column)) <= max_levels) {
      column <- factor(column)
    }
    return(column)
  })

  names(processed_data) <- names(factor_data)

  factor_data <- as.data.frame(processed_data, stringsAsFactors = FALSE)

  # Generate the model matrix for only factor variables
  if (ncol(factor_data) > 0) {
    encoded_data <- model.matrix(~ . - 1, data = factor_data)
    encoded_data <- as.data.frame(encoded_data)

    # Customize column names to reflect original name concatenated with the factor level
    names(encoded_data) <- gsub(".*\\.", "", names(encoded_data))  # Simplify names by removing prefixes
    names(encoded_data) <- gsub("(.*)\\(Intercept\\)", "\\1", names(encoded_data))  # Handle Intercept term if any
    names(encoded_data) <- gsub("(.*)$", "\\1", names(encoded_data))  # Clean up remaining names

    # Append original column names to factor levels for clarity
    for (col in names(factor_data)) {
      factor_levels <- levels(factor_data[[col]])
      for (level in factor_levels) {
        old_name <- paste0(col, level)
        new_name <- paste0(col, level)  # Customize this part if different naming convention needed
        if (old_name %in% names(encoded_data)) {
          names(encoded_data)[names(encoded_data) == old_name] <- new_name
        }
      }
    }

    # Reconstruct the full data frame with original numeric columns and encoded factors
    final_data <- cbind(numeric_data, encoded_data)
    return(final_data)
  } else {
    return(numeric_data)  # Return numeric data only if no factor columns are left
  }
}

#' Handle Response Variable
#'
#' This function preprocesses the response variable for modeling purposes. It checks the data type of the response variable and ensures it is suitable for the proposed model. If the response variable is numeric, it checks whether it is truly continuous or disguised binary. If it is categorical, it ensures it is binary with exactly two levels.
#'
#' @param y The response variable to be handled.
#' @return The processed response variable suitable for modeling.
#' @details This function checks the data type and unique values of the response variable. If the response variable is numeric, it determines whether it is continuous or disguised binary (having only 0s and 1s). If the response variable is categorical, it ensures it is binary with exactly two levels. The function returns the preprocessed response variable.
#'
#' @examples
#' # Example 1: Binary Numeric Response Variable
#' response_numeric_binary <- c(0, 1, 0, 1, 1)
#' processed_response1 <- handle_response_variable(response_numeric_binary)
#' print(processed_response1)
#'
#' @examples
#' # Example 2: Continuous Numeric Response Variable
#' response_numeric_continuous <- c(1.2, 3.5, 2.1, 4.8, 5.6)
#' processed_response2 <- handle_response_variable(response_numeric_continuous)
#' print(processed_response2)
#'
#' @examples
#' # Example 3: Binary Categorical Response Variable (2 Levels)
#' response_categorical_binary <- factor(c("yes", "no", "yes", "no", "yes"))
#' processed_response3 <- handle_response_variable(response_categorical_binary)
#' print(processed_response3)
#'
#' @examples
#' # Example 4: Binary Categorical Response Variable (More than 2 Levels)
#' response_categorical_more_than_2_levels <- factor(c("yes", "no", "maybe", "yes", "no"))
#' try(processed_response4 <- handle_response_variable(response_categorical_more_than_2_levels),silent = TRUE)
#' response_categorical_more_than_2_levels
#'
#' @examples
#' # Example 5: Unsupported Data Type for Response Variable
#' response_unsupported_data_type <- list(1, 2, 3, 4, 5)
#' try(processed_response5 <- handle_response_variable(response_unsupported_data_type),silent = TRUE)
#' response_unsupported_data_type
#'
#' @export
handle_response_variable <- function(y) {
  # Check the data type of y and the number of unique values
  unique_values <- unique(y)
  if (is.numeric(y)) {
    # Check if y is truly continuous or disguised binary (only 0s and 1s)
    if (length(unique_values) == 2) {
      # y is already a numeric binary
      if(all(unique_values %in% c(0, 1)))
      {
        return(y)
      }
      else
      {
        y <- as.numeric(factor(y)) - 1
        return(y)
      }
    }
    else {
      # y is continuous
      return(y)
    }
  } else if (is.factor(y) || is.character(y)) {
    # Categorical data handling
    if (length(unique_values) == 2) {
      y <- as.numeric(factor(y)) - 1
      return(y)
    } else {
      stop("Since the proposed model handles response categorigal as  binary and that binary variable must have exactly two levels")
    }
  } else {
    stop("Unsupported binary data type for the response variable.")
  }
}

#' Perform Pre-screening for Predictors
#'
#' This function performs pre-screening for predictors based on the number of predictors and observations. If the number of predictors is much greater than the number of observations, it prompts the user to specify the number of top K most informative predictors or defaults to 10% of the number of observations. This function performs pre-screening for feature selection based on correlation or combined method (correlation and random forest importance).
#'
#' @param X The matrix/data frame containing predictors.
#' @param y The response variable.
#' @param K (Optional) The number of top K most informative predictors to select.
#' @param res (Optional) Specify if you want to manually specify the number of top predictors.
#' @param res2 (Optional) Specify "2" to choose the combined method, otherwise the correlation method is chosen.
#' @param cor_threshold (Optional) The correlation threshold value (0-1) for filtering predictors based on correlation with the response variable (optional).
#' @return A list containing the selected predictors,preprocessed response variable, and selected predictor names.
#' @details This function checks if the number of predictors is much greater than the number of observations.
#'          If so, it prompts the user to specify the number of top K most informative predictors or defaults
#'          to 10% of the number of observations.The feature selection  allowing users to choose between correlation-based filtering, or a combination of both correlation and random forest.
#'           By selecting the most informative features, this function helps improve model performance and interpretability.
#'
#'
#' @examples
#' #Default Usage - Correlation Method
#' X <- matrix(rnorm(100), ncol = 5)
#' y <- rnorm(20)
#' prescreening(X, y)
#'
#' @examples
#' #Correlation Method with P>>n
#' X <- matrix(rnorm(100), ncol = 20)
#' y <- rnorm(5)
#' prescreening(X, y)
#' @export
prescreening <- function(X,y,K = NULL,res=NULL,res2=NULL,cor_threshold=NULL) {
  method = "1"
  if (is.null(colnames(X))) {
    colnames(X) <- paste0("X", seq_len(ncol(X)))
    cat("Column names were not set, default names X1, X2, ..., XN have been assigned.\n")
  }
  if (ncol(X) >= 2*nrow(X)){
    if(is.null(res)){
      cat("Noted that p>>n , Choose a method for feature(top predictors) selection:\n1. correlation (default)\n2. Combined Feature Selection (correlation and randomforest)\n")
      response <- readline(prompt="Enter the number for the method (1 for correlation method, 2 correlation and randomforest): ")
      method <- ifelse(tolower(response) == "2", "combined", "correlation")

      cat("The number of predictors p is much greater than the number of observations n.\n",
          "Consider specifying the K parameter to perform pre-screening for top K most informative predictors.\n")
      response <- readline(prompt="Do you want to specify the number of top K most informative predictors? (yes/no): ")
      if (tolower(response) == "yes") {
        K <- as.integer(readline(prompt="Please enter the value of K: "))
        if (K < 1 || K > ncol(X)) {
          stop("K must be between 1 and the number of columns in X.")
        }
      }
      else {
        K <- max(1, ceiling(0.1 * nrow(X)))  # Defaulting K to 10% of the number of observations
        cat("Using K =", K, "\n")
      }}
    else{
      if(tolower(res)=="no"){

        K <- max(1, ceiling(0.1 * nrow(X)))  # Defaulting K to 10% of the number of observations

      }

      method <- ifelse(tolower(res2) == "2", "combined", "correlation")


    }
  }


  y = handle_response_variable(y)

  # Execute based on chosen method
  if (!is.null(K) && K > 0){
    if (tolower(method) == "combined") {

      return(combined_feature_selection(X, y, K,cor_threshold=cor_threshold))
    } else {

      return(prescreening_correlation(X, y, K))
    }
  }
  else{
    selected_predictors <- X
    selected_predictor_names <- colnames(X)
  }
  return (list(predictors = selected_predictors,response = y, predictor_names = selected_predictor_names))
}

#' Perform feature selection based on correlation with response variable
#' SURE Independence Screening (SIS) based on Correlation
#'
#' This function performs prescreening based on correlation between predictors and response variable.
#'
#' @param X The predictor matrix (numeric matrix or data frame).
#' @param y The response variable (numeric vector).
#' @param K Number of top correlated predictors to select. If NULL, all predictors are retained.
#' @param res Reserved parameter for future use.
#' @return A list containing selected predictors, response variable, and predictor names.
#' @export
prescreening_correlation = function(X,y,K=NULL,res=NULL)
{
  # SURE Independence Screening (SIS) if K is specified
  if (!is.null(K) && K > 0) {
    # Ensure X is a numeric matrix
    X_matrix <- as.matrix(X)
    cor_matrix <- abs(cor(X_matrix, y))  # Compute absolute correlations
    top_k_indices <- order(cor_matrix, decreasing = TRUE)[1:K]  # Get indices of top K correlations
    if (length(top_k_indices) < K) {
      stop("Insufficient number of predictors to select top K.")
    }
    selected_predictors <- X_matrix[, top_k_indices, drop = FALSE]  # Ensure matrix structure is preserved
    selected_predictor_names <- colnames(X_matrix)[top_k_indices]
  } else {
    selected_predictors <- X
    selected_predictor_names <- colnames(X)
  }
  return (list(predictors = selected_predictors,response = y, predictor_names = selected_predictor_names))
}

#' Perform combined feature selection using correlation (SIS method) and Random Forest importance
#'
#' This function performs feature selection by combining correlation-based filtering (SIS method)
#' and random forest feature importance techniques.
#'
#' @param X The predictor matrix (numeric matrix or data frame).
#' @param y The response variable (numeric vector or factor).
#' @param K Number of top important features to select using random forest. If NULL, all features are retained.
#' @param cor_threshold The correlation threshold value (0-1) for filtering predictors based on correlation with the response variable.
#'
#' @return A list containing selected predictors, response variable, and predictor names.
#'
#' @export
combined_feature_selection <- function(X, y, K = NULL,cor_threshold=NULL) {
  #y = ifelse (length (unique(y))==2, as.factor(y),y)
  if(is.null(cor_threshold)){
    cor_threshold <- as.numeric(readline(prompt="Enter the correlation threshold value (0-1) or enter 0 to use deafult value as 0.1: "))
  }
  if (as.numeric(cor_threshold) == 0) {
    cat("cor_threshold not specified, using deafult value as 0.1...\n")
    corr_threshold <- 0.1
  } else {
    corr_threshold <- as.numeric(cor_threshold)
  }
  # Step 1: Correlation-based Filtering
  # Calculate the absolute correlation values
  cor_values <- abs(cor(X, y, use = "complete.obs"))
  high_corr_indices <- which(cor_values > corr_threshold)

  if (length(high_corr_indices) == 0) {
    stop("No variables meet the correlation threshold.")
  }

  # Filter features based on the correlation threshold
  filtered_X <- X[, high_corr_indices, drop = FALSE]

  # Step 2: Random Forest for Feature Importance on Filtered Features
  if (!is.null(K) && K > 0) {

    if (length(unique(y)==2))
    {
      y1 <- as.factor(y)
    }
    else
    {
      y1 <- y
    }
    # Train Random Forest on the filtered set of features
    rf_model <- randomForest(filtered_X, y1, ntree = 200, importance = TRUE)
    importance_scores <- importance(rf_model, type = 1)  # Mean decrease in Gini

    # Get indices of top K important features from the filtered list
    top_k_indices <- order(importance_scores, decreasing = TRUE)[1:K]
    selected_predictors <- filtered_X[, top_k_indices, drop = FALSE]
    selected_predictor_names <- colnames(selected_predictors)
  }
  else {
    selected_predictors <- X
    selected_predictor_names <- colnames(X)
  }

  # Return the selected features and their names
  return(list(predictors = selected_predictors, response = y, predictor_names = selected_predictor_names))
}

#' Perform Simple Regression model
#'
#' This function fits a regression model (linear and logistic) to the provided predictor matrix and response variable. It preprocesses the data by handling null values, performing pre-screening for predictors, and one-hot encoding categorical variables. The type of regression model (linear or logistic) is determined based on the nature of the response variable.
#'
#' @param X The matrix/data frame containing predictors.
#' @param y The response variable.
#' @param res (Optional) Specify if you want to manually specify the response variable.
#' @param K (Optional) The number of top most informative predictors to select.
#' @return A list containing the regression model and predictor names.
#' @details This function preprocesses the predictor variables by handling null values, performing pre-screening
#'          for top most informative predictors, and one-hot encoding categorical variables. It then fits a
#'          regression model based on the type of response variable (binary or continuous). The function returns
#'          the fitted model and predictor names.
#' @examples
#'# Example 1: Binary Response Variable
#'set.seed(123)
#'X <- matrix(rnorm(100), ncol = 10)
#'y <- c(0, 1, 0, 1, 1,1,0,1,0,1)
#'regression_result1 <- simple_regression(X, y)
#'print(summary(regression_result1$model))
#'print(regression_result1$predictors_names)
#'
#'@examples
#'# Example 2: Continuous Response Variable
#'set.seed(123)
#'X <- matrix(rnorm(100), ncol = 10)
#'y <- rnorm(20)
#'regression_result2 <- simple_regression(X, y)
#'print(summary(regression_result2$model))
#'print(regression_result2$predictors_names)
#' @export
simple_regression <- function(X, y,res=NULL,K=NULL) {
  if (is.null(colnames(X))) {
    colnames(X) <- paste0("X", seq_len(ncol(X)))
    cat("Column names were not set, default names X1, X2, ..., XN have been assigned.\n")
  }
  is_binary = ifelse((length(unique(y)) == 2),TRUE,FALSE)
  ppn = Preprocessing_nullvalues(X)
  ps = prescreening(ppn$predictors,y,res=res,K=K)
  ppe = one_hot_encode(ps$predictors)
  # Convert matrix to dataframe and add response variable
  data_for_model <- data.frame(ppe, response = ps$response)

  # Determine the model type based on the response variable
  if (is_binary) {
    final_model <- glm(response ~ ., family = binomial(), data = data_for_model)
  } else {
    final_model <- lm(response ~ ., data = data_for_model)
  }


  # Return the model coefficients and model summary
  return(list(model = final_model, predictors_names = names(ppe)))
}

#' Perform Ridge Regression Model
#'
#' This function fits a ridge regression model to the provided predictor matrix and response variable. It preprocesses the data by handling null values, performing pre-screening for predictors, and one-hot encoding categorical variables. The type of regression model (regression or classification) is determined based on the nature of the response variable.
#'
#' @param X The matrix/data frame containing predictors.
#' @param y The response variable.
#' @param lambda (Optional) The regularization parameter lambda.
#' @param res (Optional) Specify if you want to manually specify the response variable.
#' @param K (Optional) The number of top most informative predictors to select.
#' @return A list containing the ridge regression model and predictor names.
#' @details This function preprocesses the predictor variables by handling null values, performing pre-screening
#'          for top most informative predictors, and one-hot encoding categorical variables. It then fits a
#'          ridge regression model using the specified or default lambda value. The function returns the fitted
#'          model and predictor names.
#' @examples
#' # Test case 1: Numeric response variable, no pre-screening
#' X1 <- matrix(rnorm(100), nrow = 10)
#' y1 <- rnorm(10)
#' colnames(X1) <- NULL
#' result1 <- ridge_regression_model(X1, y1)
#' result1
#'
#' @examples
#' # Test case 2: Binary response variable, pre-screening with K = 5
#' X2 <- matrix(rnorm(200), nrow = 20)
#' y2 <- rbinom(20, 1, 0.5)
#' colnames(X2) <- NULL
#' result2 <- ridge_regression_model(X2, y2, K = 5)
#' result2
#'
#' @examples
#' # Test case 3: Numeric response variable, custom lambda
#' X3 <- matrix(rnorm(100), nrow = 10)
#' y3 <- rnorm(10)
#' colnames(X3) <- NULL
#' result3 <- ridge_regression_model(X3, y3, lambda = 0.5)
#' result3
#'
#' @export
ridge_regression_model <- function(X, y, lambda = NULL,res=NULL,K=NULL) {
  if (is.null(colnames(X))) {
    colnames(X) <- paste0("X", seq_len(ncol(X)))
    cat("Column names were not set, default names X1, X2, ..., XN have been assigned.\n")
  }
  is_binary = ifelse((length(unique(y)) == 2),TRUE,FALSE)
  ppn = Preprocessing_nullvalues(X)
  ps = prescreening(ppn$predictors,y,res=res,K=K)
  ppe = one_hot_encode(ps$predictors)
  predictor = as.matrix(ppe)
  family = ifelse((length(unique(y)) == 2),"binomial","gaussian")
  # Fit the final model using the determined or default lambda
  final_model <- glmnet(predictor, ps$response, alpha = 0, lambda = lambda, family = family)

  # Return the model coefficients and active predictors
  return(list(model = final_model, predictors_names = names(ppe)))
}

#' Perform Lasso Regression Model
#'
#' This function fits a Lasso regression model to the provided predictor matrix and response variable. It preprocesses the data by handling null values, performing pre-screening for predictors, and one-hot encoding categorical variables. The type of regression model (regression or classification) is determined based on the nature of the response variable.
#'
#' @param X The matrix/data frame containing predictors.
#' @param y The response variable.
#' @param lambda (Optional) The regularization parameter lambda.
#' @param res (Optional) Specify if you want to manually specify the response variable.
#' @param K (Optional) The number of top most informative predictors to select.
#' @return A list containing the lasso regression model and predictor names.
#' @details This function preprocesses the predictor variables by handling null values, performing pre-screening
#'          for top most informative predictors, and one-hot encoding categorical variables. It then fits a
#'          lasso regression model using the specified or default lambda value. The function returns the fitted
#'          model and predictor names.
#' @examples
#' ## Test case 1 : Small Dataset
#' set.seed(123)
#' # Generate sample data
#' X <- matrix(rnorm(100), ncol = 10)  # 10 numeric predictors
#' y <- sample(0:1, 10, replace = TRUE)  # Binary response variable
#' # Set lambda parameter for Lasso regression
#' lambda <- 0.1
#' # Fit Lasso regression model
#' lasso_regression_result <- lasso_regression_model(X, y, lambda = lambda)
#' lasso_regression_result
#'
#' @examples
#' # Test case 2 : Different lambda values
#' set.seed(123)
#' # Generate sample data
#' X <- matrix(rnorm(100), ncol = 10)  # 10 numeric predictors
#' y <- sample(0:1, 10, replace = TRUE)  # Binary response variable
#' # Set different lambda parameter for Lasso regression
#' lambda <- 0.2
#' # Fit Lasso regression model
#' lasso_regression_result <- lasso_regression_model(X, y, lambda = lambda)
#' lasso_regression_result
#' @export
lasso_regression_model <- function(X, y, lambda = NULL,res=NULL,K=NULL) {
  if (is.null(colnames(X))) {
    colnames(X) <- paste0("X", seq_len(ncol(X)))
    cat("Column names were not set, default names X1, X2, ..., XN have been assigned.\n")
  }
  is_binary = ifelse((length(unique(y)) == 2),TRUE,FALSE)
  ppn = Preprocessing_nullvalues(X)
  ps = prescreening(ppn$predictors,y,res=res,K=K)
  ppe = one_hot_encode(ps$predictors)
  predictor = as.matrix(ppe)
  family = ifelse((length(unique(y)) == 2),"binomial","gaussian")
  final_model <- glmnet(predictor, ps$response, alpha = 1, lambda = lambda, family = family)

  # Return the model coefficients and active predictors
  return(list(model = final_model, predictors_names = names(ppe)))
}

#' Perform Elastic Net Regression model
#'
#' This function fits an Elastic Net regression model to the provided predictor matrix and response variable. It preprocesses the data by handling null values, performing pre-screening for predictors, and one-hot encoding categorical variables. The type of regression model (regression or classification) is determined based on the nature of the response variable.
#'
#' @param X The matrix/data frame containing predictors.
#' @param y The response variable.
#' @param lambda (Optional) The regularization parameter lambda.
#' @param alpha (Optional) The mixing parameter alpha (0 for ridge, 1 for lasso).
#' @param res (Optional) Specify if you want to manually specify the response variable.
#' @param K (Optional) The number of top most informative predictors to select.
#' @return A list containing the elastic net regression model and predictor names.
#' @details This function preprocesses the predictor variables by handling null values, performing pre-screening
#'          for top most informative predictors, and one-hot encoding categorical variables. It then fits an
#'          elastic net regression model using the specified or default lambda and alpha values. The function
#'          returns the fitted model and predictor names.
#' @examples
#' ## Test case 1: Lasso Reg
#' set.seed(123)
#' # Generate sample data
#' X <- matrix(rnorm(100), ncol = 10)  # 10 numeric predictors
#' y <- sample(0:1, 10, replace = TRUE)  # Binary response variable
#' # Set lambda parameter for Lasso regression
#' lambda <- 0.1
#' # Fit Lasso regression model using elasticnet_regression_model
#' lasso_result <- elasticnet_regression_model(X, y, lambda = lambda, alpha = 1)
#' # Check for warnings
#' warnings()
#' ## In this test case, we expect a warning indicating that Lasso regression is being performed and the execution of the `elasticnet_regression_model` function is stopped.
#'
#' @examples
#' ## Test case 2 : Ridge Reg
#' set.seed(123)
#' # Generate sample data
#' X <- matrix(rnorm(100), ncol = 10)  # 10 numeric predictors
#' y <- rnorm(20)  # Continuous response variable
#' # Set lambda parameter for Ridge regression
#' lambda <- 0.1
#' # Fit Ridge regression model using elasticnet_regression_model
#' ridge_result <- elasticnet_regression_model(X, y, lambda = lambda, alpha = 0)
#' # Check for warnings
#' warnings()
#' ## In this test case, we expect a warning indicating that Ridge regression is being performed and the execution of the `elasticnet_regression_model` function is stopped.
#'
#' @examples
#' # Test case 3 : Elastic Net Regression
#' set.seed(123)
#' # Generate sample data
#' X <- matrix(rnorm(100), ncol = 10)  # 10 numeric predictors
#' y <- sample(0:1, 10, replace = TRUE)  # Binary response variable
#' # Set lambda parameter for Elastic Net regression
#' lambda <- 0.1
#' alpha <- 0.5
#' # Fit Elastic Net regression model using elasticnet_regression_model
#' elasticnet_result <- elasticnet_regression_model(X, y, lambda = lambda, alpha = alpha)
#' elasticnet_result
#' ## In this test case, Elastic Net regression is performed without any warnings.
#'
#' @export
elasticnet_regression_model <- function(X, y, lambda = NULL, alpha = 0.5,res=NULL,K=NULL) {
  if (is.null(colnames(X))) {
    colnames(X) <- paste0("X", seq_len(ncol(X)))
    cat("Column names were not set, default names X1, X2, ..., XN have been assigned.\n")
  }
  # Check for Lasso Regression
  if (alpha == 1) {
    warning("since alpha noted is 1 which fits lasso regression model,please use lasso_regression_model function and hence elasticnet_regession model execution is stopped.")
    stop("Stopping execution due to selection of lasso regression.")
  }
  # Check for Ridge Regression
  if (alpha == 0) {
    warning("since alpha noted is 0 which fits ridge regression model,please use ridge_regression_model function and hence elasticnet_regession model execution is stopped.")
    stop("Stopping execution due to selection of ridge regression.")
  }
  ppn = Preprocessing_nullvalues(X)
  ps = prescreening(ppn$predictors,y,res=res,K=K)
  ppe = one_hot_encode(ps$predictors)
  predictor = as.matrix(ppe)
  family = ifelse((length(unique(y)) == 2),"binomial","gaussian")
  # Fit the final model using the determined or default lambda
  final_model <- glmnet(predictor, ps$response, alpha = alpha, lambda = lambda, family = family)

  # Return the model coefficients and active predictors
  return(list(model = final_model, predictors_names = names(ppe)))
}

#' Train Random Forest Model
#'
#' This function trains a random forest model for classification or regression.
#'
#' @param X The matrix/data frame containing predictors.
#' @param y The response variable.
#' @param ntrees The number of trees in the random forest.
#' @return A list containing the trained random forest model and predictor names.
#' @details This function preprocesses the predictor variables by handling null values, performing pre-screening
#'          for top most informative predictors, and one-hot encoding categorical variables. It then trains a
#'          random forest model using the specified number of trees. The function supports both classification
#'          and regression tasks based on the type of response variable. The trained model and predictor names
#'          are returned in a list.
#' @examples
#'## Test case 1 : Binary Response Variable
#'set.seed(123)
#'# Generate sample data
#'X <- matrix(rnorm(100), ncol = 10)  # 10 numeric predictors
#'y <- sample(0:1, 10, replace = TRUE)  # Binary response variable
#'# Train random forest model
#'rf_result_binary <- random_forest(X, y)
#'rf_result_binary
#'
#' @examples
#'## Test Case 2: Continuous Response Variable
#'set.seed(123)
#'# Generate sample data
#'X <- matrix(rnorm(100), ncol = 10)  # 10 numeric predictors
#'y <- rnorm(20)  # Continuous response variable
#'# Train random forest model
#'rf_result_continuous <- random_forest(X, y)
#'rf_result_continuous
#'
#' @examples
#' ## Test case 3 : Custom Number of Trees
#' set.seed(123)
#' # Generate sample data
#' X <- matrix(rnorm(100), ncol = 10)  # 10 numeric predictors
#' y <- sample(0:1, 20, replace = TRUE)  # Binary response variable
#' ntrees <- 1000  # Custom number of trees
#' # Train random forest model with custom number of trees
#' rf_result_custom_ntrees <- random_forest(X, y, ntrees = ntrees)
#' rf_result_custom_ntrees
#' @export
random_forest <- function(X, y,ntrees= 500) {
  if (is.null(colnames(X))) {
    colnames(X) <- paste0("X", seq_len(ncol(X)))
    cat("Column names were not set, default names X1, X2, ..., XN have been assigned.\n")
  }
  is_binary = ifelse((length(unique(y)) == 2), TRUE, FALSE)
  ppn = Preprocessing_nullvalues(X)
  ps = prescreening(ppn$predictors, y)
  ppe = one_hot_encode(ps$predictors)
  # Convert matrix to dataframe and add response variable
  data_for_model <- data.frame(ppe, response = ps$response)

  # Train the random forest model
  if (is_binary) {
    final_model <- randomForest(factor(response) ~ ., data = data_for_model, ntree = ntrees)
  } else {
    final_model <- randomForest(response ~ ., data = data_for_model, ntree = ntrees)
  }

  # Return the trained random forest model
  return(list(model = final_model, predictors_names = names(ppe)))
}




# Interactive Regression Analysis - v8

This Streamlit app fits **linear** or **polynomial** regressions to either:
- **Group means**
- **Raw replicates**

and displays:
- **mean ± error**
- an overlaid **regression curve**
- optional **raw replicate points**
- the estimated **x_opt** from the fitted curve
- **R² (fit basis)**, **R² (means)**, and **R² (raw)**

## Main updates in v8
- The output **file name is generated automatically** from the selected Excel sheet name:
  - example: `MySheet_regression.png`
- The manual **File name** field has been removed.
- Downloaded **PNG** files are now exported with a **transparent background**.
- The app still automatically selects the **first available sheet** when an Excel file is uploaded.
- **Regression type defaults to Polynomial**.

## Run
```bash
streamlit run streamlit_regression_dual_r2_v8.py
```

## Requirements
Install dependencies first:
```bash
pip install -r requirements.txt
```

## Input Excel format
The app expects the Excel sheet to be organized like this:
- **Row 1**: numeric concentrations only
- **Rows below**: replicate values for each concentration

Example:

| 0 | 24 | 120 | 240 |
|---|---|---|---|
| rep1 | rep1 | rep1 | rep1 |
| rep2 | rep2 | rep2 | rep2 |
| rep3 | rep3 | rep3 | rep3 |

Important:
- the first row must contain only **numeric concentration values**
- replicate cells below may contain blanks, but each concentration should have data

## Meaning of the main options
### Fit based on
- **Group means**: fits one mean value per concentration
- **Raw replicates**: fits all individual experimental values

### Error bars shown on mean points
- **SD**: shows the spread of replicate values
- **SEM**: shows the uncertainty of the mean

### Weights for mean fit
Used only when fitting **Group means**:
- **None**: unweighted fit
- **SEM**: recommended when you want weights based on uncertainty of the mean
- **SD**: weights based on spread of replicates

### X-axis display
- **Numeric (true spacing)**: keeps real concentration spacing
- **Categorical (equal spacing, visual only)**: spreads concentrations evenly for readability, while the fit is still computed using the true numeric concentrations

## Meaning of the R² values
- **R² (fit basis)**: goodness of fit on the same data used to estimate the model
- **R² (means)**: how well the curve follows one mean per concentration
- **R² (raw)**: how well the same curve explains all individual replicate values

## Recommended interpretation
- Use **ANOVA/post hoc** to decide which **tested concentration** is best statistically.
- Use the regression and **x_opt** as an **interpolated estimate** between tested concentrations.
- In biological datasets, **R² (raw)** is often lower than **R² (means)** because raw replicates include real experimental variability.
- A common practical setup is:
  - show **SD** on the graph
  - fit **Group means**
  - weight the mean fit with **SEM**

## Notes
- When **Raw replicates** is selected, **Weights for mean fit** is disabled.
- When **Raw replicates** is selected, **Show raw replicate points** is automatically enabled.
- The graph includes the estimated **Max reg / x_opt** based on the fitted curve.
- **PNG** exports are transparent; **JPG** exports are not.

def create_example_model(df_red_, output):

    import pandas as pd
    from imblearn.pipeline import Pipeline
    from imblearn import FunctionSampler
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression

    # remove duplicated values
    def drop_duplicated(X,y):
        df = pd.concat([X,y], axis=1)
        df = df.drop_duplicates()
        return df.iloc[:,:-1], df.iloc[:,-1]

    DD =  FunctionSampler(func=drop_duplicated,
                          validate=False)

    # standardises all variables
    scaler = StandardScaler() 

    # here is the model we want to use.
    reg = LinearRegression()

    # create our pipeline for the data to go through.
    # This is a list of tuples with a name (useful later) and the function.
    reg_pipe = Pipeline([
        ("drop_duplicated", DD),
        ("scaler", scaler),
        ("model", reg)
    ])

    y_train = df_red_.loc[:,output]
    X_train = df_red_.drop(output, axis=1)

    return reg_pipe.fit(X_train, y_train)

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
class DropCorrelatedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, method='pearson', threshold=0.8): 
        
        """
        data_x: features to check for correlations
        method: how to calculate the correlation
        threshold: the cutoff to remove one of the correlated features
        """
        
        
        self.method = method
        self.threshold = threshold
        
    def fit(self, X, y=None):
        # make a copy
        X_ = X.copy()

        # turn to dataframe if a numpy array
        if isinstance(X_, (np.ndarray, np.generic)):
            X_ = pd.DataFrame(X_)

        # from https://www.projectpro.io/recipes/drop-out-highly-correlated-features-in-python
        cor_matrix  = X_.corr(method=self.method).abs() # get correlation and remove -
        upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > self.threshold)]
        
        # get the names of features that are corrleated
        correlated = []
        for column in upper_tri.columns:
            if any(upper_tri[column] > self.threshold):
                corr_data = upper_tri[column][upper_tri[column] > self.threshold]
                for index_name in corr_data.index:
                    correlated.append((index_name, corr_data.name))
        
        self.features_to_drop_ = np.array(to_drop)
        self.correlated_feature_sets_ = np.array(correlated)
        
        return self
        
    def transform(self, X, y=None):
        X_ = X.copy()
        
        # turn to dataframe if a numpy array
        if isinstance(X_, (np.ndarray, np.generic)):
            X_ = pd.DataFrame(X_)
        
        return X_.drop(self.features_to_drop_, axis=1)

def bike_data_prep(data):
    import pandas as pd

    data_ = data.copy()
    
    to_drop = ['casual', 'registered']
    if any(x in to_drop for x in data_.columns):
        data_.drop(['casual', 'registered'], axis=1, inplace=True)
        
    data_.rename(columns={'count':'riders', 'atemp': 'realfeel'}, inplace=True)
    data_['datetime'] = pd.to_datetime(data_['datetime'], format='%Y-%m-%d %H:%M:%S')
    data_.set_index('datetime', inplace=True)
    
    return data_


def create_example_bike_model(data_, output):

    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    
    def extract_hour(dt):
        return dt.hour
    
    def create_hour_feat(data):
        data_ = data.copy()
        data_['hour'] = data_.index.map(extract_hour)
        return data_
    
    data_ = bike_data_prep(data_)
    
    # make compatible with a scikit-learn pipeline
    hour_feat_func = FunctionTransformer(func=create_hour_feat,    # our custom function
                                         validate=False)           # prevents input being changed to numpy arrays
    
    
    hour_onehot = ColumnTransformer(
        # apply the `OneHotEncoder` to the "hour" column
        [("OHE", OneHotEncoder(drop="first"), ["hour"])],
        # don't touch all other columns, instead concatenate it on the end of the
        # changed data.
        remainder="passthrough"
    ) 
    
    lin_dummy_pipe = Pipeline([
        ("create_hour", hour_feat_func),
        ("encode_hr", hour_onehot),
        ("model", LinearRegression())
    ])

    return lin_dummy_pipe.fit(data_.drop(["temp", "riders"], axis=1), data_.loc[:,"riders"])


def example_residual_plot(dataframe, x_feat, y_feat):
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    import matplotlib.pyplot as plt

    # fit simple linear regression model
    linear_model = ols(y_feat+' ~ '+x_feat,
                       data=dataframe).fit()

    # creating regression plots
    fig = sm.graphics.plot_regress_exog(linear_model,
                                        x_feat)

    # change the plot to the plotted and fitted 
    fig.axes[0].clear()
    fig.axes[0].scatter(dataframe[x_feat], dataframe[y_feat])
    fig.axes[0].set_xlabel(x_feat)
    fig.axes[0].set_ylabel(y_feat, labelpad=15)
    fig.axes[0].set_title(x_feat + ' versus ' + y_feat, {'fontsize': 15})
    sm.graphics.abline_plot(model_results=linear_model, ax = fig.axes[0], c="black")

    # remove unneccisary plots
    fig.delaxes(fig.axes[2])
    fig.delaxes(fig.axes[2])

    # tidy label
    fig.axes[1].set_title('residuals versus '+y_feat, {'fontsize': 15})

    # residual lines on left plot
    for i, num in enumerate(dataframe[y_feat]):
        value = dataframe[y_feat][i]
        prediction = linear_model.predict(dataframe[x_feat][i:i+1])[0]
        if value > prediction:
            fig.axes[0].vlines(dataframe[x_feat][i], ymin=prediction, ymax=value, color="red", linestyles="dashed")
        else:
            fig.axes[0].vlines(dataframe[x_feat][i], ymin=value, ymax=prediction, color="red", linestyles="dashed")

    # residual lines on right plot    
    for i, num in enumerate(dataframe[y_feat]):
        resid = linear_model.resid[i]
        if 0 > resid:
            fig.axes[1].vlines(dataframe[x_feat][i], ymin=resid, ymax=0, color="red", linestyles="dashed")
        else:
            fig.axes[1].vlines(dataframe[x_feat][i], ymin=0, ymax=resid, color="red", linestyles="dashed")

    plt.show()


def plot_cv_indices(cv, X, ax, n_splits, lw=10):
    # edited from https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html#sphx-glr-auto-examples-model-selection-plot-cv-indices-py

    """Create a sample plot for indices of a cross-validation object."""
    from matplotlib.patches import Patch
    import matplotlib.pyplot as plt
    import numpy as np

    cmap_cv = plt.cm.coolwarm
    cmap_data = plt.cm.Paired

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            X.index,
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )

    # Plot the data at the top
    ax.scatter(
        X.index, [ii + 1.5] * len(X), marker="_", lw=lw, cmap=cmap_data
    )

    # Formatting
    yticklabels = list(range(n_splits))+ ["data"]
    ax.set(
        yticks=np.arange(n_splits + 1) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
    )
    plt.xticks(rotation=45, ha='right')
    ax.set_title("{}".format(type(cv).__name__), fontsize=15)
    ax.legend(
        [Patch(color=cmap_cv(0.8)), Patch(color=cmap_cv(0.02))],
        ["Testing set", "Training set"],
        loc=(1.02, 0.8),
    )
    return ax


# from Colin Rundel (http://www2.stat.duke.edu/~cr173/)
def get_coefs(m):
    """Returns the model coefficients from a Scikit-learn model object as an array,
    includes the intercept if available.
    """
    import sklearn
    import numpy as np
    
    
    # If pipeline, use the last step as the model
    if (isinstance(m, sklearn.pipeline.Pipeline)):
        m = m.steps[-1][1]
    
    
    if m.intercept_ is None:
        return m.coef_
    
    return np.concatenate([[m.intercept_], m.coef_])

# from Colin Rundel (http://www2.stat.duke.edu/~cr173/)
def model_fit(m, X, y, plot = False):
    """Returns the root mean squared error of a fitted model based on provided X and y values.
    
    Args:
        m: sklearn model object
        X: model matrix to use for prediction
        y: outcome vector to use to calculating rmse and residuals
        plot: boolean value, should fit plots be shown 
    """
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    
    y_hat = m.predict(X)
    rmse = mean_squared_error(y, y_hat, squared=False)
    
    res = pd.DataFrame(
        data = {'y': y, 'y_hat': y_hat, 'resid': y - y_hat}
    )
    
    if plot:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(121)
        sns.lineplot(x='y', y='y_hat', color="grey", data =  pd.DataFrame(data={'y': [min(y),max(y)], 'y_hat': [min(y),max(y)]}))
        sns.scatterplot(x='y', y='y_hat', data=res).set_title("Fit plot")
        
        plt.subplot(122)
        sns.scatterplot(x='y', y='resid', data=res).set_title("Residual plot")
        plt.hlines(y=0, xmin=np.min(y), xmax=np.max(y), linestyles='dashed', alpha=0.3, colors="black")
        
        plt.subplots_adjust(left=0.0)
        
        plt.suptitle("Model rmse = " + str(round(rmse, 4)), fontsize=16)
        plt.show()
    
    return rmse

# from Colin Rundel (http://www2.stat.duke.edu/~cr173/)
# TODO: Could get this working with hour?
def ridge_coef_alpha_plot(X_train, X_val, y_train, y_val):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    
    def extract_hour(dt):
        return dt.hour
    
    def create_hour_feat(data):
        data_ = data.copy()
        data_['hour'] = data_.index.map(extract_hour)
        return data_
    
    #X_train = data_prep(X_train)
    #X_val = data_prep(X_val)
    
    # make compatible with a scikit-learn pipeline
    hour_feat_func = FunctionTransformer(func=create_hour_feat,    # our custom function
                                         validate=False)           # prevents input being changed to numpy arrays
    
    
    hour_onehot = ColumnTransformer(
        # apply the `OneHotEncoder` to the "hour" column
        [("OHE", OneHotEncoder(drop="first"), ["hour"])],
        # don't touch all other columns, instead concatenate it on the end of the
        # changed data.
        remainder="passthrough"
    ) 

    scaler = ColumnTransformer(
        # apply the `StandardScaler` to the numerical data
        [("SS", StandardScaler(), ["realfeel", "humidity", "windspeed"])],
        # don't touch all other columns, instead concatenate it on the end of the
        # changed data.
        remainder="passthrough"
    ) 
    
    alphas = np.logspace(-2, 3, num=200)

    betas = [] # Store coefficients
    rmses = [] # Store validation rmses

    col_names = ['season', 'holiday', 'workingday', 'weather', 'realfeel', 'humidity', 'windspeed', 'hour']

    for a in alphas:
        m = Pipeline([
            #("create_hour", hour_feat_func),
            #("pandarizer",FunctionTransformer(lambda x: pd.DataFrame(x, columns = col_names))),
            ("scaler", scaler),
            #("pandarizer2",FunctionTransformer(lambda x: pd.DataFrame(x, columns = col_names))),
            #("encode_hr", hour_onehot),
            ("model", LinearRegression())
        ]).fit(X_train, y_train)

        # We drop the intercept as it is not included in Ridge's l2 penalty and hence not shrunk
        betas.append(get_coefs(m)[1:]) 
        rmses.append(model_fit(m, X_val, y_val))

    res = pd.DataFrame(
        data = betas,
        columns = X_train.columns # Label columns w/ feature names
    ).assign(
        alpha = alphas,
        rmse = rmses
    ).melt(
        id_vars = ('alpha', 'rmse')
    )

    sns.lineplot(x='alpha', y='value', hue='variable', data=res).set_title("Coefficients")
    plt.show()

# from Colin Rundel (http://www2.stat.duke.edu/~cr173/)
def lasso_coef_alpha_plot(X_train, X_val, y_train, y_val):
    import numpy as np
    from sklearn.pipeline import Pipeline
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import Lasso

    alphas = np.logspace(-2, 2, num=200)

    betas = [] # Store coefficients
    rmses = [] # Store validation rmses

    scaler = ColumnTransformer(
        # apply the `StandardScaler` to the numerical data
        [("SS", StandardScaler(), ["realfeel", "humidity", "windspeed"])],
        # don't touch all other columns, instead concatenate it on the end of the
        # changed data.
        remainder="passthrough"
    ) 

    for a in alphas:
        m = Pipeline([
            ("SS", scaler),
            ("Ridge", Lasso(alpha=a))]
        ).fit(X_train, y_train)
        
        # We drop the intercept as it is not included in Ridge's l2 penalty and hence not shrunk
        betas.append(get_coefs(m)[1:]) 
        rmses.append(model_fit(m, X_val, y_val))
        
    res = pd.DataFrame(
        data = betas,
        columns = X_train.columns # Label columns w/ feature names
    ).assign(
        alpha = alphas,
        rmse = rmses
    ).melt(
        id_vars = ('alpha', 'rmse')
    )

    sns.lineplot(x='alpha', y='value', hue='variable', data=res).set_title("Coefficients")
    plt.show()

# this creates the matplotlib graph to make the confmat look nicer
def pretty_confusion_matrix(confmat, labels, title, labeling=False, highlight_indexes=[]):
    import matplotlib.pyplot as plt
    import warnings
    labels_list = [["TN", "FP"], ["FN", "TP"]]
    
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            if labeling:
                label = str(confmat[i, j])+" ("+labels_list[i][j]+")"
            else:
                label = confmat[i, j]
            
            
            if [i,j] in highlight_indexes:
                ax.text(x=j, y=i, s=label, va='center', ha='center',
                        weight = "bold", fontsize=18, color='#32618b')
            else:
                ax.text(x=j, y=i, s=label, va='center', ha='center')
       
    # change the labels
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax.set_xticklabels(['']+[labels[0], labels[1]])
        ax.set_yticklabels(['']+[labels[0], labels[1]])

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    ax.xaxis.set_label_position('top')
    plt.suptitle(title)
    plt.tight_layout()
    
    plt.show()


def create_example_logmodel(X, output):

    import pandas as pd
    from imblearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from imblearn.under_sampling import RandomUnderSampler


    log_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("sampler", RandomUnderSampler(random_state=123)),
        ("model", LogisticRegression(random_state=42))])

    y_train = X.loc[:,output]
    X_train = X.drop(output, axis=1)

    return log_pipe.fit(X_train, y_train)

def clean_ufos(ufo):
    import pandas as pd
    ufo.Time = pd.to_datetime(ufo.Time, format='%m/%d/%Y %H:%M')
    ufo.set_index('Time', inplace=True)
    
    return ufo


def tidy_eu_passengers(data):
    import pycountry
    import pandas as pd
    from re import match

    # ------------
    # TIDY COLUMNS
    # ------------

    # rename columns
    airlines = data.rename({
        "geo\\time":"country",
        "tra_meas":"measurement",
        "tra_cov": "coverage"
    }, axis="columns").copy()

    non_date_cols = list(airlines.columns[0:5])

    # remove the space in column names
    airlines.columns = airlines.columns.str.replace(' ', '')

    # just get the month columns
    filtered_values = list(filter(lambda v: match('\d+M\d+', v), airlines.columns))

    # reduce columns down to years with months
    airlines = airlines[non_date_cols+filtered_values]

    # make a date column
    airlines = pd.melt(airlines,
                       id_vars=non_date_cols,
                       var_name="date",
                       value_name='vals') 

    # ---------
    # TIDY DATA
    # ---------

    # replace the 'M' with a dash
    airlines.date = airlines.date.str.replace('M', '-')

    # change to a datetime
    airlines.date = pd.to_datetime(airlines.date, format='%Y-%m')

    #set the date as the index
    airlines.set_index('date', inplace=True)

    # get a dictionary with the codes and the country name
    country_dict = {}
    for country in airlines["country"].unique():
        try:
            country_dict[country] = pycountry.countries.lookup(country).name
        except:
            pass

    # use the dictionary to replace the codes
    airlines.country = airlines.country.replace(country_dict)
    # change ":" to nan
    airlines = airlines.replace(": ", np.nan)

    # change the values to float
    airlines.vals = airlines.vals.astype("float", errors='ignore')

    # sort earliest to most recent
    airlines.sort_index(inplace=True)

    return airlines

def tidy_eu_rail_passengers(data):
    import pycountry
    import pandas as pd
    from re import match

    # ------------
    # TIDY COLUMNS
    # ------------

    # rename columns
    rail = data.rename({
        "geo\\time":"country",
    }, axis="columns").copy()

    non_date_cols = list(rail.columns[0:2])

    # remove the space in column names
    rail.columns = rail.columns.str.replace(' ', '')

    # make a date column
    rail = pd.melt(rail,
                   id_vars=non_date_cols,
                   var_name="date",
                   value_name='vals') 

    # ---------
    # TIDY DATA
    # ---------
    # format quaters using regex
    rail['date'] = rail['date'].str.replace(r'(\d+)(Q\d)', r'\1-\2', regex =True)
    # turn the quaters into dates
    rail['date'] = pd.PeriodIndex(rail['date'], freq='Q').to_timestamp()
    
    #set the date as the index
    rail.set_index('date', inplace=True)

    # get a dictionary with the codes and the country name
    country_dict = {}
    for country in rail["country"].unique():
        try:
            country_dict[country] = pycountry.countries.lookup(country).name
        except:
            pass

    # use the dictionary to replace the codes
    rail.country = rail.country.replace(country_dict)
    # change ":" to nan
    rail = rail.replace([": ", ": c", ":"], np.nan)
    # remove all spaces in values
    rail.vals = rail.vals.str.strip()

    # change the values to float
    rail.vals = rail.vals.astype("float")

    # sort earliest to most recent
    rail.sort_index(inplace=True)

    return rail
    

def plot_example_digits(x,y):
    import matplotlib.pyplot as plt
    import numpy as np

    # set random seed (change to see different examples)
    np.random.seed(seed=42)

    fig, ax = plt.subplots(1, 10, figsize=(15, 2))

    # find the unique digits
    unique_digits = np.unique(y)

    # for each unique digit...
    for digit in unique_digits:
        # find the indices for different digits
        idxs = np.where(y==digit)[0]
        # randomly select an index
        idx = np.random.choice(idxs)
        # plot the example image
        ax[digit].imshow(x.loc[idx].values.reshape(8, 8), 
                         cmap='gray', interpolation="bilinear")
        # set the subplot title as the digit
        ax[digit].set_title(digit)
        # tidy the grid up
        ax[digit].grid(False)
        ax[digit].set_xticks([])
        ax[digit].set_yticks([])

    plt.suptitle("Digit Examples")
    plt.show()

def plot_2d_clusters(x, y, ax, title=""): 
    import matplotlib.cm as cm
    import pandas as pd

    ax.set_xlim(x.min()['x'], x.max()['x'])
    ax.set_ylim(x.min()['y'], x.max()['y'])
    
    y_uniques = pd.Series(y).unique()
    colors = cm.tab10(np.linspace(0, 1, len(y_uniques)))
    for y_unique_item, c in zip(y_uniques, colors):
        x[ y == y_unique_item ].plot(
            title=title, 
            kind='scatter', 
            x='x', y='y',
            marker=f'${y_unique_item}$', 
            ax=ax, c=[c], s=40)

def k_centeroids_vis(title, data, k_max, centres=False, legend=False):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans

    for k in list(range(1, k_max+1)):
        kclust = data.copy()
        km = KMeans(n_clusters=k, random_state=1).fit(kclust)
        kclust['cluster'] = km.fit_predict(kclust)
        kclust['k']=k
        
        centeroids = pd.DataFrame(km.cluster_centers_, columns=['c1', 'c2'])
        centeroids.index.name = 'cluster'
        centeroids['k'] =k
        centeroids = centeroids.reset_index()

        if k==1:
            allkclust = kclust
            allcenteroids = centeroids
        else:
            allkclust = pd.concat([allkclust, kclust], axis=0)
            allcenteroids = pd.concat([allcenteroids,centeroids])

    fig = sns.FacetGrid(allkclust.merge(allcenteroids), col='k', hue='cluster', col_wrap=3, height=4, aspect=1)

    if centres:
        fig.map(sns.scatterplot, 'c1', 'c2', alpha=0.9, marker='o', s=40, linewidths=8,
                color='w', zorder=10, legend=False)
        fig.map(sns.scatterplot, 'c1', 'c2', marker='x', s=20, linewidths=20,
                        color='k', zorder=11, alpha=1, legend=False)
    fig.map(sns.scatterplot, 'x', 'y', alpha=0.6, s=5)

    if legend:
        fig.add_legend()
    
    plt.subplots_adjust(top=0.95)
    fig.fig.suptitle(title)    
    plt.show()


def k_means_ani(data, n_clusters=3, init = "random", n_init = 1,
                max_iter=20, random_state=1, title=None, 
                embed_limit=30000000.0):
    import pandas as pd
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from matplotlib import animation, rc
    matplotlib.rcParams['animation.embed_limit'] = embed_limit
    
    kclust = data.copy()
    
    def update_plot(i, models, scat1, scat2, scat3):
        if models[int(i/2)]==None:
            # set 1 standard color
            scat1.set_array(np.array([0]*len(kclust)))
            # makes sure the x's are not plotted
            empty_np = np.empty(models[-1].cluster_centers_.shape)
            empty_np[:]=np.NaN
            scat2.set_offsets(empty_np)
            scat3.set_offsets(empty_np)
        else:
            if (i % 2) == 0:
                # changes position
                scat2.set_offsets(models[int(i/2)].cluster_centers_)
                scat3.set_offsets(models[int(i/2)].cluster_centers_)
            else:
                # changes color
                scat1.set_array(models[int(i/2)].predict(kclust))

        return scat3,
    
    k_iters = []
    for init_seed in range(n_init):
        # just adds a break between starting with new centeroids
        [k_iters.append(None) for i in range(2)]

        for i in range(1,max_iter+1):
            km = KMeans(n_clusters=10,     # Pick a number of clusters
                        init=init,        # The initial centroids are randomly chosen as one of the data rows
                        n_init=1,         # Number of times the algorithm runs with different centroid seeds
                        algorithm="full", # Classic "Expectation Maximization" algorithm
                        max_iter=i,       # EXPLAIN
                        random_state=random_state+init_seed)
            k_iters.append(km.fit(kclust))

    numpoints = len(kclust)
    colors = k_iters[-1].predict(kclust)

    fig = plt.figure()

    scat1 = plt.scatter(kclust['x'], kclust['y'], c=colors, cmap=plt.cm.tab10, 
                      alpha=0.7)
    scat2 = plt.scatter([],[],
                      alpha=0.9, marker='o', s=40, linewidths=8,
                      color='w')
    scat3 = plt.scatter([],[],
                      marker='x', s=5, linewidths=10, c='k')
    ani = animation.FuncAnimation(fig, update_plot, 
                                  frames=range(len(k_iters)*2),
                                  fargs=(k_iters, scat1, scat2, scat3),
                                  interval=100)

    if title:
        plt.title(title)
    plt.close()
    # Note: below is the part which makes it work on Colab
    rc('animation', html='jshtml')
    return ani

def init_repeated_plot(data, max_iter=20, title = ""):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.cluster import KMeans

    inertias = []
    
    repeats = 12
    
    for k in range(1,repeats+1):
        km = KMeans(
            n_clusters=10,
            init="random",
            n_init=1,
            algorithm="full",
            max_iter=max_iter,
            random_state=k)

        kclust = data.copy()
        kclust['cluster'] = km.fit_predict(kclust)
        kclust['k']=k
        
        inertias.append(round(km.inertia_, 2))

        centeroids = pd.DataFrame(km.cluster_centers_, columns=['c1', 'c2'])
        centeroids.index.name = 'cluster'
        centeroids['k'] =k
        centeroids = centeroids.reset_index()

        if k==1:
          allkclust = kclust
          allcenteroids = centeroids
        else:
          allkclust = pd.concat([allkclust, kclust], axis=0)
          allcenteroids = pd.concat([allcenteroids,centeroids])

    fig = sns.FacetGrid(allkclust.merge(allcenteroids), col='k', hue='cluster', col_wrap=3, height=4, aspect=1)

    fig.map(sns.scatterplot, 'c1', 'c2', alpha=0.9, marker='o', s=40, linewidths=8,
            color='w', zorder=10, legend=False)
    fig.map(sns.scatterplot, 'c1', 'c2', marker='x', s=20, linewidths=20,
                    color='k', zorder=11, alpha=1, legend=False)
    fig.map(sns.scatterplot, 'x', 'y', alpha=0.6, legend=False)

    axes = fig.axes.flatten()
    
    for k in range(repeats):
        if inertias[k] == min(inertias):
            axes[k].set_title("init = "+ str(k+1) + "; SSE = "+str(inertias[k]), fontweight='bold')
        else:
            axes[k].set_title("init = "+ str(k+1) + "; SSE = "+str(inertias[k]))

    #fig.add_legend()
    plt.subplots_adjust(top=0.95)
    fig.fig.suptitle(title)  
    plt.show()

# From https://github.com/ageron/handson-ml2/blob/master/09_unsupervised_learning.ipynb
def mini_batch_img(fig_path):
    import urllib
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.datasets import load_iris
    from timeit import timeit
    from sklearn.datasets import make_blobs
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from IPython.display import Image

    np.random.seed(42)
    # this is quite computationally expensive
    if not os.path.exists(fig_path):

        blob_centers = np.array(
            [[ 0.2,  2.3],
             [-1.5 ,  2.3],
             [-2.8,  1.8],
             [-2.8,  2.8],
             [-2.8,  1.3]])
        blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])

        X, y = make_blobs(n_samples=2000, centers=blob_centers,
                          cluster_std=blob_std, random_state=7)

        def load_next_batch(batch_size):
            return X[np.random.choice(len(X), batch_size, replace=False)]

        k = 5
        n_init = 10
        n_iterations = 100
        batch_size = 100
        init_size = 500  # more data for K-Means++ initialization
        evaluate_on_last_n_iters = 10

        best_kmeans = None

        for init in range(n_init):
            minibatch_kmeans = MiniBatchKMeans(n_clusters=k, init_size=init_size)
            X_init = load_next_batch(init_size)
            minibatch_kmeans.partial_fit(X_init)

            minibatch_kmeans.sum_inertia_ = 0
            for iteration in range(n_iterations):
                X_batch = load_next_batch(batch_size)
                minibatch_kmeans.partial_fit(X_batch)
                if iteration >= n_iterations - evaluate_on_last_n_iters:
                    minibatch_kmeans.sum_inertia_ += minibatch_kmeans.inertia_

            if (best_kmeans is None or
                minibatch_kmeans.sum_inertia_ < best_kmeans.sum_inertia_):
                best_kmeans = minibatch_kmeans


        times = np.empty((100, 2))
        inertias = np.empty((100, 2))
        for k in range(1, 101):
            kmeans_ = KMeans(n_clusters=k, random_state=42)
            minibatch_kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)
            print("\r{}/{}".format(k, 100), end="")
            times[k-1, 0] = timeit("kmeans_.fit(X)", number=10, globals=globals())
            times[k-1, 1]  = timeit("minibatch_kmeans.fit(X)", number=10, globals=globals())
            inertias[k-1, 0] = kmeans_.inertia_
            inertias[k-1, 1] = minibatch_kmeans.inertia_

    # this is quite computationally expensive
    if not os.path.exists(fig_path):
        plt.figure(figsize=(10,4))

        plt.subplot(121)
        plt.plot(range(1, 101), inertias[:, 0], "r--", label="K-Means")
        plt.plot(range(1, 101), inertias[:, 1], "b.-", label="Mini-batch K-Means")
        plt.xlabel("$k$", fontsize=16)
        plt.title("Inertia", fontsize=14)
        plt.legend(fontsize=14)
        plt.axis([1, 100, 0, 100])

        plt.subplot(122)
        plt.plot(range(1, 101), times[:, 0], "r--", label="K-Means")
        plt.plot(range(1, 101), times[:, 1], "b.-", label="Mini-batch K-Means")
        plt.xlabel("$k$", fontsize=16)
        plt.title("Training time (seconds)", fontsize=14)
        plt.axis([1, 100, 0, 6])

        plt.savefig(fig_path)
        plt.show()

    else:
        display(Image(fig_path))


def DfTransformer(X, column_names = None):
    import pandas as pd
    import scipy
    import warnings

    X_ = X.copy() # so we do not alter the input data

    # turn to a pandas df if a numpy array
    if isinstance(X_, (np.ndarray, np.generic)):
        X_ = pd.DataFrame(X_, columns = column_names)

    # turn to a pandas df if sparse
    elif scipy.sparse.issparse(X_):
        X_ = pd.DataFrame.sparse.from_spmatrix(X_, columns = column_names)

    # change the column names if provided and not the same
    elif isinstance(X_, pd.DataFrame):
        if not column_names==None and set(list(X_.columns)) == set(column_names):
            X_.columns = column_names

    else:
        warnings.warn("""{} not a supported input. Input needs to be in:
        [np.ndarray, np.generic, scipy.sparse, pd.DataFrame]""".format(type(X_)))

    return X_

def to_year(series):
    import pandas as pd

    series = pd.to_datetime(series).dt.year
    return series.values.reshape(-1,1)

def rm_perc(x):
    import pandas as pd

    if isinstance(x, pd.Series):
        series = x.str.replace(r'%', '').astype(float)
        return series.values.reshape(-1,1)

    elif isinstance(x, pd.DataFrame):
        df = x.apply(lambda x_: x_.str.replace(r'%', '').astype(float))
        return df

    else:
        print("Input needs to be a Series or DataFrame")

def rem_str(series):
    term_values = {' 36 months': 36, ' 60 months': 60}
    series = series.map(term_values)
    return series.values.reshape(-1,1)

def emp_len_ext(series):
    import pandas as pd
    series = series.str.extract(r"(\d+)")
    series = pd.to_numeric(series[0])
    return series.values.reshape(-1,1)

def change_cat(series):
    term_values = {"MORTGAGE": "MORTGAGE", "RENT":"RENT", "OWN":"OWN", 'ANY': "OTHER", 'NONE': "OTHER"}
    series = series.map(term_values)
    return series.values.reshape(-1,1)


def process_lending(X_train, drop_list, feature_names = ["dti", "open_acc", 'pub_rec', 'pub_rec_bankruptcies',
                                                  'annual_inc', 'revol_bal', 'earliest_cr_line', "int_rate", 
                                                  "revol_util", "term"]):
    from sklearn.preprocessing import MinMaxScaler
    import scipy
    import warnings
    from sklearn.preprocessing import FunctionTransformer
    from feature_engine.imputation import MeanMedianImputer
    from feature_engine.outliers import ArbitraryOutlierCapper
    from sklearn.pipeline import Pipeline, make_pipeline
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import FunctionTransformer

    # adding a single entry into warnings filter
    warnings.simplefilter('error', UserWarning)

    def DfTransformer(X, column_names = None):
        X_ = X.copy() # so we do not alter the input data

        # turn to a pandas df if a numpy array
        if isinstance(X_, (np.ndarray, np.generic)):
            X_ = pd.DataFrame(X_, columns = column_names)

        # turn to a pandas df if sparse
        elif scipy.sparse.issparse(X_):
            X_ = pd.DataFrame.sparse.from_spmatrix(X_, columns = column_names)

        # change the column names if provided and not the same
        elif isinstance(X_, pd.DataFrame):
            if not column_names==None and set(list(X_.columns)) == set(column_names):
                X_.columns = column_names

        else:
            warnings.warn("""{} not a supported input. Input needs to be in:
            [np.ndarray, np.generic, scipy.sparse, pd.DataFrame]""".format(type(X_)))

        return X_

    def to_year(series):
        series = pd.to_datetime(series).dt.year
        return series.values.reshape(-1,1)

    def rm_perc(x):
        if isinstance(x, pd.Series):
            series = x.str.replace(r'%', '').astype(float)
            return series.values.reshape(-1,1)

        elif isinstance(x, pd.DataFrame):
            df = x.apply(lambda x_: x_.str.replace(r'%', '').astype(float))
            return df

        else:
            print("Input needs to be a Series or DataFrame")

    def rem_str(series):
        term_values = {' 36 months': 36, ' 60 months': 60}
        series = series.map(term_values)
        return series.values.reshape(-1,1)
    
    pre_processing = [
        # for each feature do the following....
        ("feature_transformation", ColumnTransformer([
            # ...nothing.
            ("passthrough", "passthrough", ["dti", "open_acc", 'pub_rec', 'pub_rec_bankruptcies']),
            # ...remove from the data.
            ("drop", "drop", drop_list),
            # ...log transform.
            ("log_tf", FunctionTransformer(np.log1p), ['annual_inc', 'revol_bal']),
            # ...remove the month.
            ("remove_month", FunctionTransformer(to_year), "earliest_cr_line"),
            # ...remove the "%" symbol.
            ("remove_percent", FunctionTransformer(rm_perc), ["int_rate", "revol_util"]),
            # ... remove the string "months".
            ("remove_str", FunctionTransformer(rem_str), 'term')
        ])),
        # tranform back to a dataframe
        ("DT_transform", FunctionTransformer(DfTransformer, kw_args = {"column_names":feature_names})),    
        # impute the median value for missing values for all features
        ("imputer", MeanMedianImputer(imputation_method='median')),
        # As the input needs to be a df we'll apply the capper just to dti here
        ("outlier", ArbitraryOutlierCapper(max_capping_dict={'dti': 42})),
        # normalize all the features
        ("normalization", MinMaxScaler())
    ]

    X_train_ = Pipeline(pre_processing).fit_transform(X_train)
    X_train_processed = pd.DataFrame(X_train_, columns = feature_names)

    return X_train_processed

def load_covid(UPDATE=False):
    import pandas as pd
    import os
    import re

    if UPDATE == True:
        # get the latest covid-19 UK data 
        covid_eng = pd.read_csv("https://coronavirus.data.gov.uk/api/v1/data?filters=areaType=nation;areaName=England&structure=%7B%22areaType%22:%22areaType%22,%22areaName%22:%22areaName%22,%22areaCode%22:%22areaCode%22,%22date%22:%22date%22,%22newCasesBySpecimenDate%22:%22newCasesBySpecimenDate%22,%22cumCasesBySpecimenDate%22:%22cumCasesBySpecimenDate%22,%22newFirstEpisodesBySpecimenDate%22:%22newFirstEpisodesBySpecimenDate%22,%22cumFirstEpisodesBySpecimenDate%22:%22cumFirstEpisodesBySpecimenDate%22,%22newReinfectionsBySpecimenDate%22:%22newReinfectionsBySpecimenDate%22,%22cumReinfectionsBySpecimenDate%22:%22cumReinfectionsBySpecimenDate%22%7D&format=csv")
        # save to csv
        covid_eng.to_csv("./Data/covid_rates_"+str(date.today())+".csv", index=False)

    # get a list of the data files
    onlyfiles = [f for f in os.listdir("Data") if os.path.isfile(os.path.join("Data", f))]
    # only get the covid data
    covidlist = list(filter(re.compile("covid_rates_*").match, onlyfiles))
    # load the latest data
    covid_eng = pd.read_csv("./Data/"+covidlist[-1])
    
    return covid_eng

def covid_prep(data):
    import pandas as pd
    
    # reduce data down
    covid = data[["date", "newCasesBySpecimenDate"]].copy()
    # change name of column
    covid.rename(mapper={"newCasesBySpecimenDate": "new_cases"},axis='columns', inplace=True)
    # change to datetime
    covid.date = pd.to_datetime(covid.date, format='%Y-%m-%d')
    # sort earliest to most recent
    covid = covid.sort_values(by='date')
    # Set the date to the index 
    covid = covid.set_index('date')
    
    return covid

def load_airline_passengers(UPDATE = False):

    import requests
    import pandas as pd
    import os
    import re

    if UPDATE:
        url = "https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing?file=data/avia_paoc.tsv.gz"
        filename = "./Data/"+url.split("/")[-1]
        with open(filename, "wb") as f:
            r = requests.get(url)
            f.write(r.content)
        df = pd.read_csv(filename, compression='gzip', delimiter="\t|,", engine='python')
        # save to csv
        df.to_csv("./Data/passengerOverviewCountry_"+str(date.today())+".csv", index=False)

    # get a list of the data files
    onlyfiles = [f for f in os.listdir("Data") if os.path.isfile(os.path.join("Data", f))]
    # only get the airlines data
    passCountlist = list(filter(re.compile("passengerOverviewCountry_*").match, onlyfiles))
    # load the latest data
    pass_raw = pd.read_csv("./Data/"+passCountlist[-1])
    
    return pass_raw

def load_railway_passengers(UPDATE = False):
    import requests
    import pandas as pd
    import os
    import re

    if UPDATE:
        url = "https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing?file=data/rail_pa_quartal.tsv.gz"
        filename = "./Data/"+url.split("/")[-1]
        with open(filename, "wb") as f:
            r = requests.get(url)
            f.write(r.content)
        df = pd.read_csv(filename, compression='gzip', delimiter="\t|,", engine='python')
        # save to csv
        df.to_csv("./Data/railPassengerOverviewCountry_"+str(date.today())+".csv", index=False)

    # get a list of the data files
    onlyfiles = [f for f in os.listdir("Data") if os.path.isfile(os.path.join("Data", f))]
    # only get the airlines data
    railPassCountlist = list(filter(re.compile("railPassengerOverviewCountry_*").match, onlyfiles))
    # load the latest data
    railpass_raw = pd.read_csv("./Data/"+railPassCountlist[-1])
    
    return railpass_raw

def get_auto_flight_pass_model(train, country_code, trace = False, update = False):
    import pmdarima as pm
    import os
    import pickle
    
    modelPath = os.path.join(os.path.curdir,"Model", country_code+"_sarima.pkl")
    
    if update or not os.path.exists(modelPath):
        
        autoarima_model = pm.auto_arima(np.log(train),
                                        trace=trace,  # prints the search for optimal parameters
                                        seasonal=True,  # whether our dataset displays seasonality or not
                                        m=12,  # number of observations per seasonal cycle (i.e. seasonality)
                                        d = 1, # The order of first-differencing
                                        random_state = 42 # ensures replicable results
                                       )
        # Serialize with Pickle
        with open(modelPath, 'wb') as pkl:
            pickle.dump(autoarima_model, pkl)
    else:
        # Now read it back and make a prediction
        with open(modelPath, 'rb') as pkl:
            autoarima_model = pickle.load(pkl)
            
    return autoarima_model

def plot_forecasts(model, train, test, country_code, update = False):
    # Note: Training Predictions for the flights data looks weird so I remove them
    import os
    from sklearn.metrics import mean_squared_error
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import Image
    
    imgPath = os.path.join(os.path.curdir, "Figures", country_code+"_sarima.png")
    
    if update or not os.path.exists(imgPath):
    
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # predicting the training data
        #train_forecast, train_conf_int = model.predict_in_sample(return_conf_int=True)
        # predicting the test data
        test_forecast, test_conf_int = model.predict(n_periods=len(test), return_conf_int=True)

        # update the model with the test data
        model.update(np.log(test))
        forecast_period = pd.date_range(start=test.index[-1] + pd.DateOffset(months=1),
                                        periods=24,
                                        freq="MS")
        # predicting the same number of points as our forecast_period defined above
        future_forecast, future_conf_int = model.predict(n_periods=len(forecast_period),
                                                                   return_conf_int=True)

        # add the index to the forecast in a pandas series
        future_forecast = pd.Series(future_forecast, index = forecast_period)

        title_dict = {#0: "Training Predictions",
                      0: "Test set Predictions",
                      1: "Future Forecast"
                     }

        for i, (forecast, conf_int) in enumerate([#(train_forecast, train_conf_int), 
                                                (test_forecast, test_conf_int),
                                                (future_forecast, future_conf_int),
                                               ]):

            # train plot
            axes[i].plot(train, label='train')
            axes[i].plot(test, label='test')
            axes[i].plot(np.exp(forecast), label='predictions', linestyle="--")

            # plot the confidence intervals on the forecasts
            axes[i].fill_between(forecast.index, 
                             np.exp(conf_int[:, 0]), 
                             np.exp(conf_int[:, 1]), 
                             color='k', 
                             alpha=0.1,
                             )

            axes[i].set_title(title_dict[i], fontsize=10)

        plt.legend(loc='best')

        plt.suptitle(country_code + " Airport Passengers")

        #train_rmse = np.sqrt(mean_squared_error(train, np.exp(train_forecast)))  # using np.exp() to cancel the log transformation
        test_rmse = np.sqrt(mean_squared_error(test, np.exp(test_forecast)))  # using np.exp() to cancel the log transformation
        #metrics_str = 'train RMSE: '+ str(train_rmse.round(2)) + '; test RMSE: ' + str(test_rmse.round(2))
        metrics_str = 'test RMSE: ' + str(test_rmse.round(2))
        axes[i].text(0.9,-0.2, metrics_str, size=12, ha="center", 
                     transform=axes[i].transAxes)

        fig.tight_layout()
        
        plt.savefig(imgPath)

        plt.show()
        
    else:
        display(Image(imgPath))

def prep_flights_test_train(data):
    import pandas as pd
    
    data_ = data
    # set time
    data_ = data_.loc['2003-01-01':'2019-12-01']
    # set a frequency for the data
    data_.index.freq = 'MS' # 'MS' = calendar month begin
    # create training/test set
    test_start = data_.index[-1]-pd.DateOffset(years=2)
    train = data_.loc[:test_start]
    test = data_.loc[test_start+pd.DateOffset(months=1):]
    
    return train, test
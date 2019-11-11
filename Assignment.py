import sys
import pandas as pd
import numpy as np
import statistics as stats
import scipy.stats as s
import time
from scipy.optimize import curve_fit as curve_fit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import DBSCAN, KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR, LinearSVR, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB#, ComplementNB
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

# Pre-processing
def open_process_file(filename):
    # cols = pd.read_csv(columns_filename,sep=" ",header=None)[0][0].split(',')
    data = pd.read_csv(filename, sep=",",skipinitialspace=True)

    return data


# Add columns by class
def add_columns(data, column_to_change):
    classes = data[column_to_change].unique()
    for cl in classes:
        data[cl] = np.where(data[column_to_change] == cl, 1, 0)

    return data


# Percentile filter
def percentile(data):
        if data['winPlacePerc'] >= 0.9:
            return 10
        elif data['winPlacePerc'] >= 0.8:
            return 9
        elif data['winPlacePerc'] >= 0.7:
            return 8
        elif data['winPlacePerc'] >= 0.6:
            return 7
        elif data['winPlacePerc'] >= 0.5:
            return 6
        elif data['winPlacePerc'] >= 0.4:
            return 5
        elif data['winPlacePerc'] >= 0.3:
            return 4
        elif data['winPlacePerc'] >= 0.2:
            return 3
        elif data['winPlacePerc'] >= 0.1:
            return 2
        else:
            return 1


def top_ten(data):
    if data['winPlacePerc'] >= 0.9:
        return 1
    else:
        return 0


# Plot 2 columns
def plot_from_data(data, column_y, column_x):
    x = data[column_x]
    y = data[column_y]

    plt.figure(figsize=(16, 8))
    l = column_x + ' against Placement'
    plt.plot(x, y, '+', label=l)
    data_corl_coref = np.corrcoef(x, y)
    data_corl_coref = "r = " + str(data_corl_coref[0][1].round(3))
    plt.text(0.2, 0.7, data_corl_coref, fontsize=10)
    plt.xlabel(column_x)
    plt.ylabel('Placement')
    plt.legend()
    plt.tight_layout()
    plt.show()


def get_class(data, column_name, class_name):
    data[column_name] = np.where(data.state == class_name, 1, 0)
    return data


def separate_file(raw, column_name, column_value, file_size, file_name):
    filtered_data = raw[raw[column_name] == column_value]

    # filtered_data = raw[cond1 | cond2]

    small_filtered_data = filtered_data[:file_size]

    filtered_data.to_csv(file_name, sep=',')
    small_filtered_data.to_csv('small_' + file_name, sep=',')


def data_preprocessing(file, train_test):
    raw_data = open_process_file(file)

    """Add in new and standardise columns"""
    # Remove null values - Only null value is in winPlacePerc so we remove that record
    raw_data = raw_data[raw_data.winPlacePerc.notnull()]

    # Total distance
    raw_data['totalDistance'] = raw_data['walkDistance'] + raw_data['rideDistance'] + raw_data['swimDistance']

    # Total kills
    raw_data['totalKills'] = raw_data['kills'] + raw_data['roadKills']

    # Highest kill has no distance; so delete players with distance 0 but kills > 0
    # raw_data = raw_data[((raw_data.totalKills > 0) & (raw_data.totalDistance > 0)) |
    #                     ((raw_data.totalKills == 0) & (raw_data.totalDistance == 0)) |
    #                     ((raw_data.totalKills == 0) & (raw_data.totalDistance > 0))]
    raw_data = raw_data[raw_data['totalDistance'] > 0]

    # Total items used
    raw_data['totalItems'] = raw_data['boosts'] + raw_data['heals']

    # Number of players per match
    raw_data['playersInMatch'] = raw_data.groupby('matchId')['matchId'].transform('count')

    # Placement percentiles distinct
    raw_data['placementRange'] = raw_data.apply(percentile, axis=1)
    raw_data['topTen'] = raw_data.apply(top_ten, axis=1)

    # Standardise match type
    raw_data = standardize_match_type(raw_data)

    # Small data and raw data - Small solo, duels, team, custom - Raw solo, duels, team, custom
    small_raw_data = raw_data[:500000]
    small_raw_data.to_csv(train_test+ '_small.csv', sep=',')
    raw_data.to_csv(train_test+ '.csv', sep=',')

    # Separate files based on match type
    separate_file(raw_data, 'matchType', 'Solo', 500000, train_test + '_solo.csv')
    separate_file(raw_data, 'matchType', 'Duo', 500000, train_test + '_duel.csv')
    separate_file(raw_data, 'matchType', 'Squad', 500000, train_test + '_squad.csv')


def standardize_match_type(data):
    data['matchType'][data['matchType'] == 'normal-solo'] = 'Solo'
    data['matchType'][data['matchType'] == 'solo'] = 'Solo'
    data['matchType'][data['matchType'] == 'solo-fpp'] = 'Solo'
    data['matchType'][data['matchType'] == 'normal-solo-fpp'] = 'Solo'
    data['matchType'][data['matchType'] == 'normal-duo-fpp'] = 'Duo'
    data['matchType'][data['matchType'] == 'duo'] = 'Duo'
    data['matchType'][data['matchType'] == 'normal-duo'] = 'Duo'
    data['matchType'][data['matchType'] == 'duo-fpp'] = 'Duo'
    data['matchType'][data['matchType'] == 'squad'] = 'Squad'
    data['matchType'][data['matchType'] == 'squad-fpp'] = 'Squad'
    data['matchType'][data['matchType'] == 'normal-squad'] = 'Squad'
    data['matchType'][data['matchType'] == 'normal-squad-fpp'] = 'Squad'
    data['matchType'][data['matchType'] == 'flaretpp'] = 'Custom'
    data['matchType'][data['matchType'] == 'flarefpp'] = 'Custom'
    data['matchType'][data['matchType'] == 'crashtpp'] = 'Custom'
    data['matchType'][data['matchType'] == 'crashfpp'] = 'Custom'

    return data


def normalise_data_by_player_count(data):
    data['killPlace'] = data['killPlace'] * ((100 - data['playersInMatch']) / 100 + 1)
    data['boosts'] = data['boosts'] * ((100 - data['playersInMatch']) / 100 + 1)
    data['walkDistance'] = data['walkDistance'] * ((100 - data['playersInMatch']) / 100 + 1)
    data['weaponsAcquired'] = data['weaponsAcquired'] * ((100 - data['playersInMatch']) / 100 + 1)

    data['kills'] = data['kills'] * ((100 - data['playersInMatch']) / 100 + 1)
    data['damageDealt'] = data['damageDealt'] * ((100 - data['playersInMatch']) / 100 + 1)
    data['boosts'] = data['boosts'] * ((100 - data['playersInMatch']) / 100 + 1)
    data['heals'] = data['heals'] * ((100 - data['playersInMatch']) / 100 + 1)
    data['rideDistance'] = data['rideDistance'] * ((100 - data['playersInMatch']) / 100 + 1)
    data['swimDistance'] = data['swimDistance'] * ((100 - data['playersInMatch']) / 100 + 1)

    return data


def insights(plot):
    # raw_solo = open_process_file('./data/train.csv')
    # raw_solo = open_process_file('./data/small_train_solo.csv')
    # raw_solo = open_process_file('./data/train_duel.csv')
    raw_solo = open_process_file('./data/train_Squad.csv')

    """Solo insights"""

    # PLAYERS JOINED INSIGHTS
    plt.figure(figsize=(16, 8))

    pd.value_counts(raw_solo['playersInMatch'], ascending=True).plot.bar(label='playersInMatch')
    if plot:
        plt.show()

    # The data depends on the amount of players in a match so ideally we would normalise by the playersInMatch
    # raw_solo = normalise_data_by_player_count(raw_solo)

    raw_solo = raw_solo.drop(columns=[ 'Id', 'groupId', 'matchId','maxPlace','numGroups','rankPoints','winPoints', 'Unnamed: 0', 'teamKills'])

    # CORRELATION TO WIN RATES
    plt.figure(figsize=(16, 8))

    corr = raw_solo.corr()
    show_corr = corr.copy()
    show_corr[show_corr.abs() < 0.4] = False

    sns.heatmap(corr,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values, annot=show_corr)
    if plot:
        plt.show()
    # KILLS INSIGHTS
    plt.figure(figsize=(16, 8))
    raw_solo['killRange'] = pd.cut(raw_solo['totalKills'], [-1, 0, 2, 5, 10, 100],
                                    labels=['0 kills', '1-2 kills', '2-4 kills', '5-10 kills', '>10 kills'])

    sns.boxplot(x='killRange', y='winPlacePerc', data=raw_solo, palette='Set3', saturation=0.8, linewidth=2.5)
    # sns.boxplot(x='killRange', y='placementRange', data=raw_solo, palette='Set3', saturation=0.8, linewidth=2.5)
    plt.xlabel("Total Kills")
    plt.ylabel("Placement Percentage")
    plt.title("Kills against Placement")

    if plot:
        plt.show()

    # KILL PLACEMENT DISTRIBUTION
    plt.figure(figsize=(16, 8))

    x = raw_solo['placementRange']

    sns.distplot(x,kde_kws={'bw':1})

    if plot:
        plt.tight_layout()
        plt.show()

    # KILL PLACEMENT DISTRIBUTION QQ
    # plt.figure(figsize=(16, 8))
    #
    # measurements = x
    # s.probplot(measurements, dist="norm", plot=plt)
    #
    # if plot:
    #     plt.tight_layout()
    #     plt.show()

    # PLAYER WITH THE HIGHEST KILLS

    plt.figure(figsize=(16, 8))

    x = raw_solo['totalKills']

    # the histogram of the data
    # plt.hist(x, density=True, facecolor='g', label='totalKills')
    sns.distplot(x,kde_kws={'bw':1})

    if plot:
        plt.tight_layout()
        plt.show()

    # totalKills QQ
    # plt.figure(figsize=(16, 8))
    #
    # measurements = x
    # s.probplot(measurements, dist="norm", plot=plt)
    #
    # if plot:
    #     plt.tight_layout()
    #     plt.show()

    highest_kills = raw_solo.iloc[raw_solo['totalKills'].idxmax()].totalKills
    highest_kills_distance = raw_solo.iloc[raw_solo['totalKills'].idxmax()].totalDistance
    highest_kills_headshots = raw_solo.iloc[raw_solo['totalKills'].idxmax()].headshotKills
    winning_players = raw_solo[raw_solo['placementRange'] == 10]
    raw_solo['headshotAccuracy'] = (raw_solo['headshotKills']/raw_solo['kills'])*100
    headshot_100 = raw_solo[raw_solo['headshotAccuracy'] == 100]
    headshot_nnan = raw_solo[raw_solo['headshotAccuracy'].notnull()]

    print('The player with the highest kills has {}, got {} headshots and traveled a total of {}m.'
          .format(highest_kills,highest_kills_headshots, highest_kills_distance))

    print("On average players get {:.2f} kills, while 90% Players get {} kills"
          .format( np.mean(raw_solo.totalKills.values), (raw_solo.totalKills.quantile(0.9))))

    print("On average players in the top 10% get {:.2f} kills, while 90% them get {:.2f} kills."
          .format( np.mean(winning_players.totalKills.values), (winning_players.totalKills.quantile(0.9))))

    # HIGHEST HEADSHOT %

    print('Several players have a headshot accuracy of 100%, the best being {}/{} headshots'
          .format(headshot_100.sort_values(by=['kills'], ascending=False).iloc[0]['totalKills'],
                  headshot_100.sort_values(by=['kills'], ascending=False).iloc[0]['totalKills']))

    plt.figure(figsize=(16, 8))

    x = headshot_nnan['headshotAccuracy']

    # the histogram of the data
    # plt.hist(x, density=True, facecolor='g', label='totalKills')
    sns.distplot(x)

    if plot:
        plt.tight_layout()
        plt.show()

    # KILL PLACEMENT

    plt.figure(figsize=(16, 8))

    x = raw_solo['killPlace']

    # plt.hist(x, density=True, facecolor='g')
    sns.distplot(x)

    if plot:
        plt.show()
        plot_from_data(raw_solo, 'winPlacePerc', 'killPlace')

    print("On average players in the top 10% are placed {:.2f} in the kills ranking, while 90% them are placed {:.2f}."
          .format( np.mean(winning_players.killPlace.values), (winning_players.killPlace.quantile(0.9))))

    # Kill placement QQ
    plt.figure(figsize=(16, 8))

    measurements = x
    s.probplot(measurements, dist="norm", plot=plt)

    if plot:
        plt.tight_layout()
        plt.show()

    # totalDistance

    plt.figure(figsize=(16, 8))

    x = raw_solo['walkDistance']

    # the histogram of the data
    # plt.hist(x, density=True, facecolor='g')
    sns.distplot(x,kde_kws={'bw':1})
    if plot:
        plt.show()
        plot_from_data(raw_solo, 'winPlacePerc', 'walkDistance')

    # Walk distance QQ
    plt.figure(figsize=(16, 8))

    measurements = x
    s.probplot(measurements, dist="norm", plot=plt)

    if plot:
        plt.tight_layout()
        plt.show()
    print("On average players travel {:.2f}m, while 90% players travel {:.2f}m."
          .format( np.mean(raw_solo.walkDistance.values), (raw_solo.walkDistance.quantile(0.9))))

    print("On average players in the top 10% travel {:.2f}m, while 90% them travel {:.2f}m."
          .format( np.mean(winning_players.walkDistance.values), (winning_players.walkDistance.quantile(0.9))))


    # ITEMS USED
    plt.figure(figsize=(16, 8))
    x = raw_solo['weaponsAcquired']
    # the histogram of the data
    sns.distplot(x,kde_kws={'bw':1})
    # plt.hist(x, density=True, facecolor='g')

    if plot:
        plt.show()
        plot_from_data(raw_solo, 'winPlacePerc', 'weaponsAcquired')

    # Weapons aquired QQ
    # plt.figure(figsize=(16, 8))
    #
    # measurements = x
    # s.probplot(measurements, dist="norm", plot=plt)
    #
    # if plot:
    #     plt.tight_layout()
    #     plt.show()

    plt.figure(figsize=(16, 8))
    x2 = raw_solo['boosts']
    sns.distplot(x2,kde_kws={'bw':1})
    # plt.hist(x2, density=True, facecolor='g')

    if plot:
        plt.show()
        plot_from_data(raw_solo, 'winPlacePerc', 'boosts')

    # boosts QQ
    plt.figure(figsize=(16, 8))

    measurements = x2
    s.probplot(measurements, dist="norm", plot=plt)

    if plot:
        plt.tight_layout()
        plt.show()

    print("On average players use {:.2f} items, while 90% players use {:.2f}."
          .format( np.mean(raw_solo.totalItems.values), (raw_solo.totalItems.quantile(0.9))))

    print("On average players in the top 10% use {:.2f} items, while 90% them use {:.2f}."
          .format( np.mean(winning_players.totalItems.values), (winning_players.totalItems.quantile(0.9))))

    # raw_solo['placementRP'] = raw_solo['placementRange'].value_counts() / raw_solo['placementRange'].count()


def classifications():

    """ BASELINE """
    # raw_solo_b = pd.read_csv('./data/train_V2.csv', sep=",", skipinitialspace=True)
    # raw_solo_b = raw_solo_b[raw_solo_b.winPlacePerc.notnull()]
    # raw_solo_b['placementRange'] = raw_solo_b.apply(percentile, axis=1)
    # raw_solo_b_1 = raw_solo_b['matchType'] == 'solo'
    # raw_solo_b_2 = raw_solo_b['matchType'] == 'solo-fpp'
    # raw_solo_b = raw_solo_b[raw_solo_b_1 + raw_solo_b_2]
    #
    # raw_train_solo_b = raw_solo_b[:100000]
    # raw_test_solo_b = raw_solo_b[100000:110000]
    #
    # x_b = raw_train_solo_b[['killPlace', 'weaponsAcquired', 'walkDistance']]
    # y_b = raw_train_solo_b['winPlacePerc']
    #
    # x_test_b = raw_test_solo_b[['killPlace', 'weaponsAcquired', 'walkDistance']]
    # y_train_b = raw_test_solo_b['winPlacePerc']
    #
    # multi_linear_regression(x_b, y_b, x_test_b, y_train_b)

    """ MULTI-LINEAR REGRESSION """
    # raw_solo = open_process_file('./data/train.csv')
    # raw_solo = open_process_file('./data/train_solo.csv')
    # raw_solo = open_process_file('./data/train_duel.csv')
    raw_solo = open_process_file('./data/train_squad.csv')

    # raw_solo = normalise_data_by_player_count(raw_solo)

    # plt.figure(figsize=(16, 8))
    # sns.pairplot(raw_solo,  hue="placementRange", vars=['totalDistance','killPlace','boosts'])
    #
    # plt.show()
    # exit()
    # Balanced data
    top_tens = raw_solo[raw_solo['topTen'] == 1]
    others = raw_solo[raw_solo['topTen'] == 0]
    others = others[:len(top_tens)]
    balanced = top_tens.append(others)
    balanced = shuffle(balanced)


    # print("Predicting exact player placement")
    # for clf in REGRESSIONS:
    #     kfold(raw_solo, clf, 10, 'winPlacePerc')

    # print()
    print("Predicting player placement range (1st-10th, 10th-20th etc)")
    for clf in CLASSIFIERS:
        kfold(raw_solo, clf, 10, 'placementRange')
    for clf in REGRESSIONS:
        kfold(raw_solo, clf, 10, 'placementRange')
    #
    # print()
    # print("Predicting binary 'In top ten' or 'Not in top ten'")
    # print('Imbalanced')
    # for clf in CLASSIFIERS:
    #     kfold(raw_solo, clf, 10, 'topTen')
    #
    # print()
    print('Balanced')
    for clf in CLASSIFIERS:
        kfold(balanced, clf, 10, 'topTen')


def confusion_analysis():
    # raw_solo = open_process_file('/data/train.csv')
    raw_solo = open_process_file('/data/train_solo.csv')
    # raw_solo = open_process_file('./data/train_duel.csv')
    # raw_solo = open_process_file('./data/train_squad.csv')

    print()
    print('Confusion matrices')

    raw_train_solo = raw_solo[:int(len(raw_solo)*0.6)]
    raw_test_solo = raw_solo[int(len(raw_solo)*0.6):]

    x = raw_train_solo[['killPlace', 'weaponsAcquired', 'walkDistance']]
    x_test = raw_test_solo[['killPlace', 'weaponsAcquired', 'walkDistance']]

    y = raw_train_solo['topTen']
    y_test = raw_test_solo['topTen']

    # clf = ComplementNB()
    # y_pred = clf.fit(x, y).predict(x_test)
    # print('ComplimentNB')
    # print(confusion_matrix(list(y_test), y_pred))

    clf = GaussianNB()
    y_pred = clf.fit(x, y).predict(x_test)
    print('GaussianNB')
    print(confusion_matrix(list(y_test), y_pred))

    clf = RandomForestRegressor(random_state=0, n_estimators=10)
    y_pred = clf.fit(x, y).predict(x_test)
    print('RandomForestRegressor')
    print(confusion_matrix(list(y_test), y_pred.astype(int)))

    clf = KNeighborsClassifier(101)
    y_pred = clf.fit(x, y).predict(x_test)
    print('KNeighborsClassifier')
    print(confusion_matrix(list(y_test), y_pred))

    clf = DecisionTreeClassifier(random_state=0)
    y_pred = clf.fit(x, y).predict(x_test)
    print('DecisionTreeClassifier')
    print(confusion_matrix(list(y_test), y_pred))


def multi_linear_regression(x, y, x_test, y_test):
    """
        Create and fit modal - make sure its through 0,0
    """
    model = LinearRegression(fit_intercept=True)
    model.fit(x, y) # Creates co-efficients
    """
        Print coefficients and print function
    """
    m = model.coef_
    print(m)
    coeff_df = pd.DataFrame(m, x.columns, columns=['Coefficient'])
    r_squared = model.score(x_test, y_test)
    intercept = model.intercept_
    # print("r = {}".format(float(r_squared) ** 0.5))

    pred = model.predict(x_test)
    mean_absolute_error = get_error_rate(list(pred), list(y_test))
    print('Mean absolute error {}'.format(mean_absolute_error))
    return r_squared


def complement_nb(x, y, x_test, y_test):
    # USING SCORE IS A BIT AWKS In multi-label classification, this is the subset accuracy which is a harsh metric
    # since you require for each sample that each label set be correctly predicted.

    clf = ComplementNB()
    y_pred = clf.fit(x, y).predict(x_test)
    print(clf.feature_log_prob_)
    accuracy = accuracy_score(y_test, y_pred)
    y_pred = clf.fit(x, y).predict(x_test)
    print(confusion_matrix(list(y_test), y_pred.astype(int)))
    return accuracy


def gaussian_nb(x, y, x_test, y_test):
    clf = GaussianNB()
    y_pred = clf.fit(x, y).predict(x_test)
    mean_accuracy = clf.score(x_test, y_test)
    print(clf.theta_)

    accuracy = accuracy_score(y_test, y_pred)
    # print('accuracy {}'.format(accuracy))
    y_pred = clf.fit(x, y).predict(x_test)
    print(confusion_matrix(list(y_test), y_pred.astype(int)))
    return accuracy


def random_forrest(x, y, x_test, y_test):
    """
        Create and fit modal - make sure its through 0,0
    """
    model = RandomForestRegressor(random_state=0, n_estimators=10)

    model.fit(x, y) # Creates co-efficients
    """
        Print coefficients and print function
    """
    r_squared = model.score(x_test, y_test)
    print(model.feature_importances_)
    pred = model.predict(x_test)
    mean_absolute_error = get_error_rate(list(pred), list(y_test))
    print('Mean absolute error {}'.format(mean_absolute_error))

    return r_squared


def random_forrest_c(x, y, x_test, y_test):
    clf = RandomForestClassifier(random_state=0, n_estimators=10)
    clf.fit(x, y)
    # print("accuracy: {} ".format(acc))
    y_pred = clf.fit(x, y).predict(x_test)
    mean_accuracy = clf.score(x_test, y_test)
    print(clf.feature_importances_)

    accuracy = accuracy_score(y_test, y_pred)
    # print('accuracy {}'.format(accuracy))
    y_pred = clf.fit(x, y).predict(x_test)
    print(confusion_matrix(list(y_test), y_pred.astype(int)))
    return accuracy


def lin_svm(x, y, x_test, y_test):
    clf = LinearSVR(random_state=0, tol=1e-5)
    clf.fit(x, y)
    acc = clf.score(x_test,y_test)
    # print("accuracy: {} ".format(acc))

    return acc


def logistic_reg(x, y, x_test, y_test):
    clf = LogisticRegression(random_state=0, tol=1e-5)
    clf.fit(x, y)
    acc = clf.score(x_test,y_test)
    # print("accuracy: {} ".format(acc))
    # print(clf.coef_)
    y_pred = clf.fit(x, y).predict(x_test)
    print(confusion_matrix(list(y_test), y_pred.astype(int)))
    return acc


def svr(x, y, x_test, y_test):
    clf = SVR(gamma='auto')
    clf.fit(x, y)
    acc = clf.score(x_test,y_test)
    # print("accuracy: {} ".format(acc))
    return acc


def svc(x, y, x_test, y_test):
    clf = SVC(gamma='auto', decision_function_shape='ovo')
    clf.fit(x, y)
    acc = clf.score(x_test,y_test)
    # print("accuracy: {} ".format(acc)
    return acc


def knnc(x, y, x_test, y_test):
    clf = KNeighborsClassifier(101)
    clf.fit(x, y)
    # print("accuracy: {} ".format(acc))
    y_pred = clf.fit(x, y).predict(x_test)
    mean_accuracy = clf.score(x_test, y_test)
    accuracy = accuracy_score(y_test, y_pred)
    # print('accuracy {}'.format(accuracy))
    y_pred = clf.fit(x, y).predict(x_test)
    print(confusion_matrix(list(y_test), y_pred.astype(int)))
    return accuracy


def decision_tree(x, y, x_test, y_test):
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(x, y)
    y_pred = clf.fit(x, y).predict(x_test)
    mean_accuracy = clf.score(x_test, y_test)
    # print(clf.feature_importances_)
    y_pred = clf.fit(x, y).predict(x_test)
    print(confusion_matrix(list(y_test), y_pred.astype(int)))
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def get_error_rate(predic, expect):
    forecast_errors = [expect[i] - predic[i] for i in range(len(expect))]

    forecast_errors_abs = [abs(forecast_errors[i]) for i in range(len(forecast_errors))]
    mean_absolute_error = stats.mean(forecast_errors_abs)
    return mean_absolute_error


CLASSIFIERS = [
    random_forrest_c,
    gaussian_nb,
    complement_nb,
    logistic_reg,
    knnc,
    decision_tree,
]

REGRESSIONS = [
    multi_linear_regression,
    random_forrest
]


def kfold(data, classifier, test_s, target):
    kf = KFold(n_splits=test_s)
    kf.get_n_splits(data)
    accuracy = 0
    for train_index, test_index in kf.split(data):
        raw_train_solo = data.iloc[train_index]
        raw_test_solo = data.iloc[test_index]

        x = raw_train_solo[['killPlace', 'boosts', 'walkDistance', 'weaponsAcquired']]
        x_test = raw_test_solo[['killPlace', 'boosts', 'walkDistance','weaponsAcquired']]

        y = raw_train_solo[target]
        y_test = raw_test_solo[target]

        accuracy += classifier(x, y, x_test, y_test)
        avg_accuracy = accuracy/test_s

    print('{} KFolds Accuracy for {} : {}'.format(test_s, str(classifier.__name__), avg_accuracy))
    return avg_accuracy


def clustering():
    # raw_solo = open_process_file('./data/train.csv')
    raw_solo = open_process_file('/data/train_solo.csv')
    # raw_solo = open_process_file('./data/train_duel.csv')
    # raw_solo = open_process_file('./data/train_squad.csv')
    raw_solo = normalise_data_by_player_count(raw_solo)
    y = raw_solo[['placementRange']]
    x = raw_solo[['kills','damageDealt','weaponsAcquired','boosts','heals', 'walkDistance','rideDistance','swimDistance']]

    # plt.figure(figsize=(16, 8))

    # KMEANS
    kmeans = KMeans(n_clusters=4, random_state=0).fit(x)
    print(kmeans.cluster_centers_)

    # EM
    gmm = GaussianMixture(n_components=5, covariance_type='full').fit(x)
    print(gmm.means_)
    print(gmm.weights_)
    print(gmm.covariances_)


    # DBSCAN
    clustering = DBSCAN(eps=3, min_samples=2).fit(x)
    print(clustering.components_)
    print(clustering.core_sample_indices_)

    # SILOHUETTE


if __name__ == "__main__":
    file_name = sys.argv[1]

    # raw_data = open_process_file(file_name)
    # raw_data['totalDistance'] = raw_data['walkDistance'] + raw_data['rideDistance'] + raw_data['swimDistance']
    #
    # print(raw_data[(raw_data['totalDistance'] == 0) & (raw_data['kills'] > 0)])

    # temp = raw_solo[(raw_solo.matchType != 'solo') & (raw_solo.matchType != 'solo-fpp') &
    #                 (raw_solo.matchType != 'normal-solo-fpp') & (raw_solo.matchType != 'normal-solo') &
    #                 (raw_solo.matchType != 'duo') & (raw_solo.matchType != 'duo-fpp') &
    #                 (raw_solo.matchType != 'normal-duo-fpp') & (raw_solo.matchType != 'normal-duo') &
    #                 (raw_solo.matchType != 'squad') & (raw_solo.matchType != 'squad-fpp') &
    #                 (raw_solo.matchType != 'normal-squad-fpp') & (raw_solo.matchType != 'normal-squad')]
    # print(temp['matchType'])

    # raw_data = raw_data[((raw_data.totalKills > 0) & (raw_data.totalDistance > 0)) |
    #                     ((raw_data.totalKills == 0) & (raw_data.totalDistance == 0)) |
    #                     ((raw_data.totalKills == 0) & (raw_data.totalDistance > 0))]

    insights(True)
    # classifications()
    # confusion_analysis()
    # clustering()

    # data_preprocessing(file_name, 'train')
    # exit()
    # data_preprocessing(file_name, 'test')
    # exit()
    # raw_d = open_process_file('./data/small_train_Solo.csv')
    # raw = open_process_file(./data/train_solo.csv')
    # raw['first_place'] = np.where(raw.winPlacePerc == 1.0, 1, 0)

    # raw['cluster'] = list(zip(raw.killPlace, raw.winPlacePerc))
    # raw['cluster'] = raw[['killPlace', 'winPlacePerc']].values.tolist()
    # print(raw['cluster'])


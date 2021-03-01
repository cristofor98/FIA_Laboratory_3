# Linear regression model
> This is  a model based on data about apartaments sales that can predict apartment price

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [How to run](#setup)
* [Code Examples](#features)
* [Contact](#contact)

## General info
Based on a txt file with data about apartament price  and with help of linear regression  I build a model that if you give him data that is in table he can predict the price of apartment

## Technologies
* Python 3.8.2
* Pandas 1.2.2
* scikit-learn 0.24.1
* seaborn 0.11.1
* matplotlib 3.3.4
* numpy 1.20.1

## How to run
$ python linear_regression.py

## Code Examples
Rule example: 

    from sklearn.linear_model import LinearRegression
    lm = LinearRegression()
    lm.fit(x_train,y_train)
    print('Intercept :', round(lm.intercept_,2))
    print('Slope :', round(lm.coef_[0],2))
    coeff_df = pd.DataFrame(lm.coef_,x.columns,columns=['Coefficient'])
    predictions = lm.predict(x_test)


## Contact
Created by @cristofor98

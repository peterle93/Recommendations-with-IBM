from rec_app import app
import json
import plotly
import pandas as pd
# The home directory is the location recommendation_app.py that run app.
# so import Recommender need add recommendation directory
from recommendation.recommender import Recommender
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Scatter


#app = Flask(__name__)  #This should not be used, or app will use this and index.html cannot be found.

rec=Recommender()


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    import pdb
    #pdb.set_trace()
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    ## 1st plot data
    latent_factors_num,test_accuracy,train_accuracy=rec.mf_calculate_error()
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Scatter(
                    x=latent_factors_num,
                    y=train_accuracy,
                    mode = 'lines',
                    name = 'train_accuracy'
                ),
                Scatter(
                    x=latent_factors_num,
                    y=test_accuracy,
                    mode = 'lines',
                    name = 'test_accuracy'
                )
            ],
            'layout': {
                'title': 'Accuracy vs. Number of Latent Factors',
                'yaxis': {
                    'title': "Accuracy Rate"
                },
                'xaxis': {
                    'title': "Number of Latent Factors"
                }
            }
        }
    ]
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    articles = rec.recommend_articles(int(query))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        articles_result=articles
    )


def main():
    app.run(host='127.0.0.1', port=3001, debug=True)


if __name__ == '__main__':
    main()

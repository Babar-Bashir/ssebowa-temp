import secrets
import asyncio
import os.path
import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
from shutil import copyfile
from flask import *
from ssebowa.preprocessing import generic_preprocessing as gp
from ssebowa.modules import logistic as lg
from ssebowa.modules import naive_bayes as nb
from ssebowa.modules import linear_svc as lsvc
from ssebowa.modules import knn
from ssebowa.modules import decision_tree as dtree
from ssebowa.modules import random_forest as rfc
from ssebowa.visualization import visualize as vis
from ssebowa.nocache import nocache
from ssebowa import app

global posted
global eda 
global profile
save_path = "ssebowa/uploads/"
exts = ["csv", "json", "yaml"]
posted = 0
profile = ""
eda=0


@app.route("/")
@app.route("/preprocess", methods=["GET", "POST"])
@nocache
def preprocess():
    eda=0
    
    if request.method == "POST":
        
        if request.form["Submit"] == "Upload":
            data = request.files["data"]
            ext = data.filename.split(".")[1]
            if ext in exts:
                session["ext"] = ext
                session["fname"] = data.filename
                data.save("ssebowa/uploads/" + data.filename)
                df = gp.read_dataset("ssebowa/uploads/" + data.filename)
                df.to_csv("ssebowa/clean/clean.csv")
                session["haha"] = True
                flash(f"File uploaded successully", "success")
            else:
                flash(f"Upload Unsuccessful. Please try again", "danger")

        elif request.form["Submit"] == "Delete":
            try:
                df = gp.read_dataset("ssebowa/clean/clean.csv")
                df = gp.delete_column(df, request.form.getlist("check_cols"))
                df.to_csv("ssebowa/clean/clean.csv", mode="w", index=False)
                #flash(f"Column(s) deleted Successfully", "success")
            except:
                flash(f"Error! Upload Dataset", "danger")


        elif request.form["Submit"] == "Clean":
            try:
                df = gp.read_dataset("ssebowa/clean/clean.csv")
                print(request.form["how"])
                if request.form["how"] != "any":
                    df = gp.treat_missing_numeric(
                        df, request.form.getlist("check_cols"), how=request.form["how"]
                    )
                elif request.form["howNos"] != None:
                    df = gp.treat_missing_numeric(
                        df,
                        request.form.getlist("check_cols"),
                        how=float(request.form["howNos"]),
                    )

                df.to_csv("ssebowa/clean/clean.csv", mode="w", index=False)
                flash(f"Column(s) cleant Successfully", "success")
            except:
                flash(f"Error! Upload Dataset", "danger")

        elif request.form["Submit"] == "Visualize":
            global posted
            df = gp.read_dataset("ssebowa/clean/clean.csv")

            x_col = request.form["x_col"]

            if vis.hist_plot(df, x_col):
                posted = 1
        
        elif request.form["Submit"] == "EDA":
            print("eda1")
            
            df = gp.read_dataset("ssebowa/clean/clean.csv")
            profile = ProfileReport(df, title="Ssebowa.ai",explorative=True)
            profile.to_file("ssebowa/templates/ssebowa.html")
            eda=1
            return render_template('ssebowa.html')
            
    
    print("[Routes->before session.get(haha)] : ",session.get("haha"))
    if("ext" in session and os.path.exists("ssebowa/clean/clean.csv")):
    #if session.get("haha") != None and session.get("haha") !=False:
        df = gp.read_dataset("ssebowa/clean/clean.csv")
        if(df != -110):
            description = gp.get_description(df)
            columns = gp.get_columns(df)
            print(columns)
            dim1, dim2 = gp.get_dim(df)
            head = gp.get_head(df)

            return render_template(
                "preprocess.html",
                active="preprocess",
                title="Preprocess",
                filename=session["fname"],
                posted=posted,
                eda=eda,
                no_of_rows=len(df),
                no_of_cols=len(columns),
                dim=str(dim1) + " x " + str(dim2),
                description=description.to_html(
                    classes=[
                        "table-bordered",
                        "table-striped",
                        "table-hover",
                        "thead-light",
                    ]
                ),
                columns=columns,
                head=head.to_html(
                    classes=[
                        "table",
                        "table-bordered",
                        "table-striped",
                        "table-hover",
                        "thead-light",
                    ]
                ),
            )
        else:
            return render_template(
                "preprocess.html", active="preprocess", title="Preprocess",
            )
    else:
        return render_template(
            "preprocess.html", active="preprocess", title="Preprocess",
        )


@app.route("/classify", methods=["GET", "POST"])
def classify():
    acc = 0
    if request.method == "POST":
        target = request.form["target"]
        gp.arrange_columns(target)
        classifier = int(request.form["classifier"])
        hidden_val = int(request.form["hidden"])
        scale_val = int(request.form["scale_hidden"])
        encode_val = int(request.form["encode_hidden"])
        columns = vis.get_columns()

        if hidden_val == 0:
            data = request.files["choiceVal"]
            ext = data.filename.split(".")[1]
            if ext in exts:
                data.save("uploads/test." + ext)
            else:
                return "File type not accepted!"
            choiceVal = 0
        else:
            choiceVal = int(request.form["choiceVal"])

        if classifier == 0:
            ret_vals = lg.logisticReg(choiceVal, hidden_val, scale_val, encode_val)
            if hidden_val == 0 or hidden_val == 1:
                return render_template(
                    "classifier_page.html",
                    acc=ret_vals[0],
                    report=[
                        ret_vals[1].to_html(
                            classes=[
                                "table",
                                "table-bordered",
                                "table-striped",
                                "table-hover",
                                "thead-light",
                            ]
                        )
                    ],
                    conf_matrix=[
                        ret_vals[2].to_html(
                            classes=[
                                "table",
                                "table-bordered",
                                "table-striped",
                                "table-hover",
                                "thead-light",
                            ]
                        )
                    ],
                    choice=hidden_val,
                    classifier_used=classifier,
                    active="classify",
                    title="Classify",
                )
            elif hidden_val == 2:
                return render_template(
                    "classifier_page.html",
                    acc=ret_vals[0],
                    report=ret_vals[1],
                    conf_matrix=ret_vals[2],
                    choice=hidden_val,
                    classifier_used=classifier,
                    active="classify",
                    title="Classify",
                    cols=columns,
                )

        elif classifier == 1:
            ret_vals = nb.naiveBayes(choiceVal, hidden_val, scale_val, encode_val)
            if hidden_val == 0 or hidden_val == 1:
                return render_template(
                    "classifier_page.html",
                    acc=ret_vals[0],
                    report=[
                        ret_vals[1].to_html(
                            classes=[
                                "table",
                                "table-bordered",
                                "table-striped",
                                "table-hover",
                                "thead-light",
                            ]
                        )
                    ],
                    conf_matrix=[
                        ret_vals[2].to_html(
                            classes=[
                                "table",
                                "table-bordered",
                                "table-striped",
                                "table-hover",
                                "thead-light",
                            ]
                        )
                    ],
                    choice=hidden_val,
                    classifier_used=classifier,
                    active="classify",
                    title="Classify",
                    cols=columns,
                )
            elif hidden_val == 2:
                return render_template(
                    "classifier_page.html",
                    acc=ret_vals[0],
                    report=ret_vals[1],
                    conf_matrix=ret_vals[2],
                    choice=hidden_val,
                    classifier_used=classifier,
                    active="classify",
                    title="Classify",
                    cols=columns,
                )

        elif classifier == 2:
            ret_vals = lsvc.lin_svc(choiceVal, hidden_val, scale_val, encode_val)
            if hidden_val == 0 or hidden_val == 1:
                return render_template(
                    "classifier_page.html",
                    acc=ret_vals[0],
                    report=[
                        ret_vals[1].to_html(
                            classes=[
                                "table",
                                "table-bordered",
                                "table-striped",
                                "table-hover",
                                "thead-light",
                            ]
                        )
                    ],
                    conf_matrix=[
                        ret_vals[2].to_html(
                            classes=[
                                "table",
                                "table-bordered",
                                "table-striped",
                                "table-hover",
                                "thead-light",
                            ]
                        )
                    ],
                    choice=hidden_val,
                    classifier_used=classifier,
                    active="classify",
                    title="Classify",
                    cols=columns,
                )
            elif hidden_val == 2:
                return render_template(
                    "classifier_page.html",
                    acc=ret_vals[0],
                    report=ret_vals[1],
                    conf_matrix=ret_vals[2],
                    choice=hidden_val,
                    classifier_used=classifier,
                    active="classify",
                    title="Classify",
                    cols=columns,
                )

        elif classifier == 3:

            scale_val = 1
            ret_vals = knn.KNearestNeighbours(
                choiceVal, hidden_val, scale_val, encode_val
            )
            if hidden_val == 0 or hidden_val == 1:
                return render_template(
                    "classifier_page.html",
                    acc=ret_vals[0],
                    report=[
                        ret_vals[1].to_html(
                            classes=[
                                "table",
                                "table-bordered",
                                "table-striped",
                                "table-hover",
                                "thead-light",
                            ]
                        )
                    ],
                    conf_matrix=[
                        ret_vals[2].to_html(
                            classes=[
                                "table",
                                "table-bordered",
                                "table-striped",
                                "table-hover",
                                "thead-light",
                            ]
                        )
                    ],
                    choice=hidden_val,
                    classifier_used=classifier,
                    active="classify",
                    title="Classify",
                    cols=columns,
                )
            elif hidden_val == 2:
                return render_template(
                    "classifier_page.html",
                    acc=ret_vals[0],
                    report=ret_vals[1],
                    conf_matrix=ret_vals[2],
                    choice=hidden_val,
                    classifier_used=classifier,
                    active="classify",
                    title="Classify",
                    cols=columns,
                )

        elif classifier == 4:
            ret_vals = dtree.DecisionTree(choiceVal, hidden_val, scale_val, encode_val)
            if hidden_val == 0 or hidden_val == 1:
                return render_template(
                    "classifier_page.html",
                    acc=ret_vals[0],
                    report=[
                        ret_vals[1].to_html(
                            classes=[
                                "table",
                                "table-bordered",
                                "table-striped",
                                "table-hover",
                                "thead-light",
                            ]
                        )
                    ],
                    conf_matrix=[
                        ret_vals[2].to_html(
                            classes=[
                                "table",
                                "table-bordered",
                                "table-striped",
                                "table-hover",
                                "thead-light",
                            ]
                        )
                    ],
                    choice=hidden_val,
                    classifier_used=classifier,
                    active="classify",
                    title="Classify",
                    cols=columns,
                )
            elif hidden_val == 2:
                return render_template(
                    "classifier_page.html",
                    acc=ret_vals[0],
                    report=ret_vals[1],
                    conf_matrix=ret_vals[2],
                    choice=hidden_val,
                    classifier_used=classifier,
                    active="classify",
                    title="Classify",
                    cols=columns,
                )
        elif classifier == 5:
            ret_vals = rfc.RandomForest(choiceVal, hidden_val, scale_val, encode_val)
            if hidden_val == 0 or hidden_val == 1:
                return render_template(
                    "classifier_page.html",
                    acc=ret_vals[0],
                    report=[
                        ret_vals[1].to_html(
                            classes=[
                                "table",
                                "table-bordered",
                                "table-striped",
                                "table-hover",
                                "thead-light",
                            ]
                        )
                    ],
                    conf_matrix=[
                        ret_vals[2].to_html(
                            classes=[
                                "table",
                                "table-bordered",
                                "table-striped",
                                "table-hover",
                                "thead-light",
                            ]
                        )
                    ],
                    choice=hidden_val,
                    classifier_used=classifier,
                    active="classify",
                    title="Classify",
                    cols=columns,
                )
            elif hidden_val == 2:
                return render_template(
                    "classifier_page.html",
                    acc=ret_vals[0],
                    report=ret_vals[1],
                    conf_matrix=ret_vals[2],
                    choice=hidden_val,
                    classifier_used=classifier,
                    active="classify",
                    title="Classify",
                    cols=columns,
                )
    elif request.method == "GET":
        columns = vis.get_columns()
        return render_template(
            "classifier_page.html", active="classify", title="Classify", cols=columns
        )


@app.route("/clear", methods=["GET"])
def clear():
    session.clear()
    return redirect("/")


@app.route("/visualize", methods=["GET", "POST"])
@nocache
def visualize():
    if request.method == "POST":
        x_col = request.form["x_col"]
        y_col = request.form["y_col"]

        df = vis.xy_plot(x_col, y_col)
        heights = np.array(df[x_col]).tolist()
        weights = np.array(df[y_col]).tolist()

        newlist = []
        for h, w in zip(heights, weights):
            newlist.append({"x": h, "y": w})
        ugly_blob = str(newlist).replace("'", "")

        columns = vis.get_columns()
        print(x_col)
        return render_template(
            "visualize.html",
            cols=columns,
            src="img/pairplot1.png",
            xy_src="img/fig.png",
            posted=1,
            data=ugly_blob,
            active="visualize",
            x_col_name=str(x_col),
            y_col_name=str(y_col),
            title="Visualize",
        )

    else:
        vis.pair_plot()
        columns = vis.get_columns()
        return render_template(
            "visualize.html",
            cols=columns,
            src="img/pairplot1.png",
            posted=0,
            active="visualize",
            title="Visualize",
        )


@app.route("/col.csv")
@nocache
def col():
    return send_file("visualization/col.csv", mimetype="text/csv", as_attachment=True)


@app.route("/pairplot1.png")
@nocache
def pairplot1():
    return send_file(
        "static/img/pairplot1.png", mimetype="image/png", as_attachment=True
    )


@app.route("/tree.png")
@nocache
def tree():
    return send_file("static/img/tree.png", mimetype="image/png", as_attachment=True)

@app.route("/eda")
@nocache
def renderEda():
    return render_template('ssebowa.html')
    #return send_file("templates/ssebowa.html", mimetype="image/png", as_attachment=True)

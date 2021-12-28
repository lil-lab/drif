TEMPLATE_HTML = """
<html>
  <head>
    <style>
    body {background-color: #dddddd;}
    .container {background-color: #ffffff; margin: 5px; padding: 5px;  box-shadow: 0 2px 4px 0 rgba(0, 0, 0, 0.2), 0 3px 10px 0 rgba(0, 0, 0, 0.19);}
    .correct-predictions-container {display: block; float:left; }
    .wrong-predictions-container {display: block; float:left; }
    .per-obj-mistakes-container {overflow: scroll; }
    td { padding: 10px;}
    th, td {border: 1px solid #dddddd; }
    img { height: 64px; }
    </style>
  </head>
  <body>
    <div class="summary-container container">
        <h3>Object Pairwise Matching Results</h3>
        <p># Examples: ${TOTAL_COUNT}$</p>
        <p># Correct: ${TOTAL_CORRECT}$</p>
        <p># Mistakes: ${TOTAL_WRONG}$</p>
        <p>Accuracy: ${ACCURACY}$</p>
        </hr>
        <p>Cumulative # Examples: ${CUM_COUNT}$</p>
        <p>Cumulative # Correct: ${CUM_CORRECT}$</p>
        <p>Cumulative # Mistakes: ${CUM_MISTAKES}$</p>
        <p>Cumulative Accuracy: ${CUM_ACCURACY}$</p>
    </div>
    <div class="per-obj-mistakes-container container">
      <h3>Per-Object Mistakes</h3>
      <table>
        <tr>
          <th>Object ID</th>
          <th>Overall Accuracy</th>
          ${OTHER_OBJECT_IDS}$
        </tr>
        ${OBJECT_MISTAKE_ROWS}$
      </table>
    </div>
    <div class="correct-predictions-container container">
      <h3>Correct Predictions</h3>
      <table>
        <tr>
          <th>A</th>
          <th>A'</th>
          <th>C</th>
        </tr>
        ${CORRECT_PREDICTIONS_ROWS}$
      </table>
    </div>
    <div class="wrong-predictions-container container">
      <h3>Wrong Predictions</h3>
      <table>
        <tr>
          <th>A</th>
          <th>A'</th>
          <th>C</th>
        </tr>
        ${WRONG_PREDICTIONS_ROWS}$
      </table>
    </div>
  </body>
</html>
"""

ROW_HTML_TEMPLATE = """
<tr>
    <td><img src="${SRC_A}$"/></td>
    <td><img src="${SRC_B}$"/></td>
    <td><img src="${SRC_C}$"/></td>
</tr>
<tr>
    <td>Reference</td>
    <td>Score: ${SCORE_B}$</td>
    <td>Score: ${SCORE_C}$</td>
</tr>
"""

OBJECT_MISTAKE_ROW_TEMPLATE = """
<tr>
  <td>${OBJ_ID}$</td>
  <td>${OBJ_ACC}$</td>
  ${OTHER_OBJECT_ACC}$
</tr>
"""


def fill(html, token, inner_html):
    full_tok = "${" + str(token) + "}$"
    idx = html.find(full_tok)
    html_out = html[:idx] + str(inner_html) + html[idx+len(full_tok):]
    return html_out


def multi_fill(html: object, tokens: object, inner_htmls: object) -> object:
    for token, inner_html in zip(tokens, inner_htmls):
        html = fill(html, token, inner_html)
    return html

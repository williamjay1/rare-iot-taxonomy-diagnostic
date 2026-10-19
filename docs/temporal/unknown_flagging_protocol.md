# Frozen prediction chronology diagnostic

This post review secondary analysis was specified after the ten label closed set
F1 results and label support were inspected, but before unknown flagging scores
were computed. It adds no model fitting, changes no time boundary, and chooses
no threshold on assessment data.

For each of the eleven frozen arms and three fitting seeds, define an unknown
row as a true label absent from the fitted label encoder. Use one minus the
maximum fitted class probability as the unknown score. Report AUROC and average
precision, unknown prevalence, and per seed values. Average precision has unknown
prevalence as its constant score reference. These metrics assess flagging, not
recognition of the individual unseen attack labels or mean rare class recall.

Audit the saved probability simplex, row identifiers, fitted class order, and
labels against the completed chronology records. Scores use the saved float32
probabilities without a refit. Three seeds share one future period, so their
standard deviation is model variation, not independent temporal replication.

Report every arm, including reversals. Do not choose an operating threshold or
claim a deployment false alarm rate from this assessment.

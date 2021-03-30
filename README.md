
![Banner](images/Banner.png)

# capstone2


## Introduction:
Breast Cancer is the most common cancer maligancy in women around the world and is curable in 70-80% of patients with early detection.  In the U.S. alone, roughly 2.1 million individuals are affected.  The development of gene microarray technologies has allowed the detection of gene expression amoung thousands of genes simultanously.  Technological insights could help patient outcomes as well as save costs in non-effective treatments. This technology in concert with the migration to electronic health records can now be utilitzed to help understand the underlying pathways and outcomes of various disease states and patient outcomes.



<!-- Beast cancer has previously been classifed based on tumor type (ductal, lobular infiltrating carcinoa, etc), HER2 receptor status, histological grade and others.  Recenlty, with cheaper, faster, and more abundant sequencing technology, the possibility of gene expression profiling (GEP) has loomed as a possible diagnostic tool. __With the complex nature of biological pathways, machine learning and big data could be the tool to elucidate the obscure pathways that have not been discovered. -->

*in silica solutions faster and more effective that laboratory processing
*could save costs of treatment if high risk individuals are identified early and a proactive treament plan can be made.
*breast cancer data could provide insights into other forms of cancer

In this study we will be trying to find underlying features in 


## Exploratory Data Analysis

The Molecular Taxonomy of Breast Cancer International Consortium (METABRIC) is an organization hosting a database  with clinical and genetic data from 1,904 primary breast cancer samples.  Genes are listed as feature columns and Z-scores of rRNA expression levels are listed as values. In this study, we will be focusing on the mRNA expression levels



![EDA dataframe](images/EDA_df.png)

Clinical information such as treatment and age at diagnosis are also features.

![Age at diagnosis](images/age_at_diagnosis_density.png)

Of the 1,904 patients, 622 were classfied as deceased due to cancer. and as a group had a lower time of survival after diagnosis.

![cancer death probabilty](images/probability_density_cancer_death_non_cancer_death.png)


![mutation count](images/Mutation_count.png)

![survival plot](images/kmf_survival_plt.png)

![classification report](images/classification_report_of_models_v1.png)




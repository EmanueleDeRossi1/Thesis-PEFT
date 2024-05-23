\subsubsection{UDAPTER}

The idea of using adapters for domain adaptation has been proposed previously. In UDAPTER \cite{udapter}, two methods for enhancing parameter efficiency in unsupervised domain adaptation (UDA) using pre-trained language models are presented. The architecture of the two methods is borrowed from \citet{adapter-fusion}. 

The first method, called Two-Steps Domain and Task Adapter (TS-DT Adapter) consists of two separate adapters: a \textit{domain adapter}, which is used to learn domain-invariant features, and a \textit{task adapter}, that learns the task. The two adapters are trained separately, in a two-step manner. First the domain adapter is trained to generate domain-invariant representations (Fig. \ref{fig:udapter} a), and then the task adapter is stacked on top and trained separately, while the domain adapter is left frozen (Fig. \ref{fig:udapter} b). The loss function of the domain adapter is calculated by using MK-MMD \cite{mk-mmd}, a function that calculates the distribution discrepancy between the probability distribution of the source and target data by comparing the means of samples from different distributions using multiple kernels. The task adapter stacked on top of the domain adapter is used to learn task-specific representations, and the loss is calculated using cross-entropy loss between the source label and the source prediction.  

The second method, called Joint Domain Task Adapter uses a single adapter instead of two. This adapter (Fig. \ref{fig:udapter} c) learns domain-invariant and task-specific representations jointly. The trade-off between the two losses is then regularized by an adaptation factor  $\lambda$.

\begin{figure}[h]

    \includegraphics[width=0.5\textwidth]{figures/udapter.png}
    \caption{The two UDAPTER architectures: Two-Steps Domain and Task Adapter (\textit{a} and \textit{b}) and Joint Domain Task Adapter (\textit{c})}
    \label{fig:udapter}
\end{figure}


The two methods are then evaluated on two different tasks: sentiment analysis, using the Multi Domain Sentiment Analysis Dataset \cite{mdsad}, and natural language inference, using the Multigenre Natural Language Inference (MNLI) corpus \cite{mnli}. Each dataset has 5 domains, which results in 20 domain adaptation scenarios per dataset. In our project, though, we won't consider these datasets, since we are interested in the task of Entity Resolution.

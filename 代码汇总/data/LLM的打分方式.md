设公司为 \(i\)，周为 \(t\)。

对公司 \(i\) 在第 \(s\) 周的所有新闻，记新闻集合为：

\[ \mathcal{N}_{i,s} \]

每条新闻 \(n\) 有一个大模型风险分：

\[ r_{i,n} \]

先定义公司 \(i\) 在第 \(s\) 周的新闻总风险分：

\[ S_{i,s}=\sum_{n \in \mathcal{N}_{i,s}} r_{i,n} \]

以及该周新闻条数：

\[ C_{i,s}=|\mathcal{N}_{i,s}| \]

如果某周没有新闻，则：

\[ S_{i,s}=0,\quad C_{i,s}=0 \]

当前脚本对公司 \(i\) 在本周 \(t\) 的年度回望分数定义为：

\[ A_{i,t} = \begin{cases} \frac{\sum_{k=0}^{51} S_{i,t-k}}{\sum_{k=0}^{51} C_{i,t-k}}, & \text{如果 } \sum_{k=0}^{51} C_{i,t-k} > 0 \\ 0, & \text{如果 } \sum_{k=0}^{51} C_{i,t-k} = 0 \end{cases} \]

也就是说，本周分数 \(A_{i,t}\) 是：

\[ \text{本周及过去51周的新闻风险分总和} \div \text{本周及过去51周的新闻总条数} \]

展开写就是：

\[ A_{i,t} = \frac{ S_{i,t}+S_{i,t-1}+\cdots+S_{i,t-51} }{ C_{i,t}+C_{i,t-1}+\cdots+C_{i,t-51} } \]

注意这里包含本周 \(t\)。

如果进一步展开到新闻层面：

\[ A_{i,t} = \frac{ \sum_{k=0}^{51} \sum_{n \in \mathcal{N}_{i,t-k}} r_{i,n} }{ \sum_{k=0}^{51} |\mathcal{N}_{i,t-k}| } \]

所以它本质上是：**公司 \(i\) 在本周及过去51周内所有新闻 `risk_score` 的算术平均值**。

之后，脚本把所有公司、所有周中非零的 \(A_{i,t}\) 放在一起，计算全市场中位数：

\[ M = \operatorname{median}\{A_{i,t}: A_{i,t}>0\} \]

最后生成标签：

\[ Label_{i,t} = \begin{cases} 0, & A_{i,t}=0 \\ 1, & 0 < A_{i,t} \le M \\ 2, & A_{i,t} > M \end{cases} \]
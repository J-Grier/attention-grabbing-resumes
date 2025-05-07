# ✨ Attention-Grabbing Resumes: Conditional Text Generation for Customized Applications

## Overview

This project leverages advanced deep learning techniques—specifically Conditional Generative Adversarial Networks (CGANs) with attention mechanisms—to automatically generate targeted, industry-specific resume content. By conditioning on specific job roles or sectors, the model produces contextually relevant resume bullet points that help applicants tailor their resumes effectively and quickly for multiple job opportunities.

---

## 🎯 Objectives

- **Primary Goal**: Automatically generate high-quality, industry-targeted resume bullet points using conditional generative modeling.
- **Practical Impact**: Reduce the time and effort required for applicants to tailor their resumes for various roles, significantly enhancing application efficiency and effectiveness.
- **Technical Innovation**: Demonstrate advanced generative modeling and attention mechanisms to produce coherent, contextually-aware textual content.

---

## 🛠️ Methods and Architecture

### Conditional Generative Adversarial Networks (CGANs)

- **Generator Network**: Takes industry/job-category labels and generates coherent, relevant resume bullet points.
- **Discriminator Network**: Evaluates realism and relevance, explicitly conditioning outputs on industry/job labels to guide meaningful generation.

### Attention Mechanisms

- Explicitly integrates attention layers into the generator to improve textual coherence and ensure key industry-specific details are highlighted clearly.
- Provides enhanced interpretability and contextual accuracy.

---

## 📐 Model Workflow

1. **Data Preparation**
   - Gathered and processed a curated dataset of high-quality resume examples spanning multiple industries (Tech, Finance, Agriculture).
   - Implemented custom preprocessing scripts to clean and structure textual data.

2. **CGAN with Attention Training**
   - Designed and implemented CGAN architecture with explicit attention layers.
   - Leveraged PyTorch for training, optimization, and performance tuning.
   - Conducted systematic hyperparameter tuning for optimal performance.

3. **Evaluation and Validation**
   - Generated resume text conditioned explicitly on specified industry labels.
   - Evaluated qualitative coherence, realism, and industry relevance through manual review and quantitative evaluation metrics (BLEU, ROUGE).

---

## 🧪 Prompt vs CGAN Output (Failure Case Gallery)

The CGAN was conditioned on simple prompts reflecting job titles and industries. Below are raw outputs from the model, revealing key failure patterns that motivate the shift to an Attention-based architecture.

<div align="center"> <table> <tr> <th>Prompt: "Senior Accountant in Finance"</th> <th>Prompt: "Software Developer in Tech"</th> <th>Prompt: "Farmer in Agriculture"</th> </tr> <tr> <td><img src="images/CGANFinance.png" alt="Finance Resume" width="300"/></td> <td><img src="images/CGANSD.png" alt="Tech Resume" width="300"/></td> <td><img src="images/CGANFarmer.png" alt="Agriculture Resume" width="300"/></td> </tr> </table> </div>

<details>
<summary>🧾 Actual Prompt Texts Used</summary>

<pre><code># Finance (used with category_name = 'Finance')
prompt_text = (
    "Professional experience in Finance:\n"
    "Senior Accountant with extensive experience in accounting, financial analysis, and management.\n"
    "Experience includes:\n"
)

# Tech (no explicit category label)
prompt_text = (
    "Professional experience in Tech:\n"
    "Software Developer skilled in Python, machine learning, NLP, and cloud technologies:\n• "
)

# Agriculture (no explicit category label)
prompt_text = (
    "Professional experience in Agriculture:\n"
    "Farmer skilled in crop management, livestock care, sustainable farming, and equipment maintenance:\n• "
)
</code></pre>

</details>

---

## 🚧 Limitations & Future Directions

### Current Limitations
- Limited training data per industry reduces potential diversity and detail.
- Requires manual validation to ensure high-quality outputs.

### Future Enhancements
- Expand dataset significantly for enhanced generalizability and diversity.
- Integrate large pre-trained language models (e.g., GPT-based) to further boost realism and contextual accuracy.
- Implement interactive user interfaces (e.g., Streamlit) to allow easy real-time resume customization and export.

---

## 🚀 Recent Updates & Experimentation (April 2025)

- Successfully integrated explicit attention mechanisms within CGAN architecture, significantly improving the coherence and contextual relevance of generated resume bullets.
- Enhanced industry-conditioned performance, validated through qualitative and quantitative assessments.

---

## ✅ Conclusion

This project demonstrates the practical value and efficiency of combining conditional generative modeling and attention mechanisms for resume customization. By generating targeted, industry-specific resume content, this approach streamlines the application process, allowing job seekers to quickly produce tailored, high-impact resumes. Future developments will focus on further enhancing realism, usability, and adaptability across a broader range of professional fields.

---
## 📚 References

- Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661). *Advances in Neural Information Processing Systems*, 2672-2680.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). [Attention is All You Need](https://arxiv.org/abs/1706.03762). *Advances in Neural Information Processing Systems*, 5998-6008.

---
## Author

**John Grier** 
MS Data Science Candidate, Illinois Tech
GitHub: J-Grier

*Note: The main modeling notebook (`Attention_CGAN_Resume_Generator.ipynb`) is not included in this repository. For access, please contact the author directly.*

---

## 📂 Project Structure
```
attention-grabbing-resumes/
│
├── Notebooks/
│   └── Attention_CGAN_Resume_Generator.ipynb     # Main model implementation and experiments
│
├── Utils/
│   └── preprocess.py                             # Custom preprocessing functions
│
├── README.md                                     # Project overview, instructions, and summary
├── requirements.txt                              # Environment dependencies
└── data/                                         # Training datasets and examples
```

## 🛠️ Setup and Requirements

### Requirements
See `requirements.txt` for all package dependencies. Typical requirements include:

```bash
torch
torchvision
numpy
pandas
scikit-learn
nltk


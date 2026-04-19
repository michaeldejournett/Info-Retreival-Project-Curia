# IEEE Paper Template - Information Retrieval Project

This folder contains an IEEE-formatted paper template for the Curia Information Retrieval project.

## Contents

- **main.tex** - Main LaTeX document with IEEE conference format
- **references.bib** - BibTeX bibliography file with sample citations
- **Makefile** - Build automation for PDF generation
- **README.md** - This file

## Structure

### Paper Sections

The template includes the following standard IEEE sections:

1. **Abstract & Keywords** - Brief summary and relevant keywords
2. **Introduction** - Problem statement and contributions
3. **Related Work** - Literature review and positioning
4. **Methodology** - Technical approach and methods
   - System Architecture
   - Local Model Support
   - Query Expansion
   - Event Management and Search
5. **Experiments** - Evaluation setup, results, and analysis
6. **Conclusion** - Summary and future directions
7. **References** - Bibliography

## Building the Paper

### Requirements

- `pdflatex` - LaTeX distribution
- `bibtex` - Bibliography tool

### Commands

```bash
# Build PDF
make

# Build and view PDF
make view

# Clean build artifacts
make clean

# Show help
make help
```

## Customization

Before finalizing, update:

1. **Author Information** (lines 21-24 in main.tex)
   - Author name(s)
   - Institution
   - Email address

2. **Title** (line 19)
   - Replace with your actual paper title

3. **Abstract** (lines 29-31)
   - Write your abstract (150-250 words typical)

4. **Content Sections**
   - Fill in methodology details
   - Add experimental results and tables
   - Include figures and graphs

5. **Bibliography** (references.bib)
   - Add your actual citations
   - Use standard BibTeX format

## Adding Figures and Tables

### Figures
```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{path/to/figure.png}
    \caption{Figure description}
    \label{fig:label}
\end{figure}
```

### Tables
```latex
\begin{table}[h]
    \centering
    \caption{Table caption}
    \begin{tabular}{lcc}
        \toprule
        Column1 & Column2 & Column3 \\
        \midrule
        Data1 & Data2 & Data3 \\
        \bottomrule
    \end{tabular}
    \label{table:label}
\end{table}
```

## Cross-References

Use labels and references to link sections, figures, and tables:

```latex
\label{sec:methodology}
\ref{sec:methodology}
\cite{citation_key}
```

## Tips

- Keep line length reasonable (60-80 characters) for readability
- Use descriptive labels for figures and tables
- Include citations for all external claims
- Test compilation frequently during writing
- Maintain consistent notation and terminology

## IEEE Format Notes

- Conference format (8.5" x 11", two columns)
- Standard font sizes and spacing
- Figure and table captions follow IEEE conventions
- References use IEEE citation style

## Resources

- [IEEE Xplore](https://ieeexplore.ieee.org/) - For finding references
- [Overleaf](https://www.overleaf.com/) - Online LaTeX editor
- [LaTeX Documentation](https://www.latex-project.org/help/) - LaTeX help

## Output

Running `make` will generate:
- **main.pdf** - Final compiled paper

Good luck with your paper!

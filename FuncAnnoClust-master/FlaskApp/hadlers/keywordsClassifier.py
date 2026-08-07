import polars as pl
import re


def keywordsClassify(function, data):
    if function in data.processed_kw_functions:
        return 0

    if function in data.kwCls_functions_set:
        data.processed_kw_functions.add(function)
        return 0

    kwClf = data.getKwCls()
    count = 0

    func_lower = function.lower()

    def add_kw(category, system, subsystem):
        nonlocal count
        kwClf.extend(pl.DataFrame({
            'Category': category,
            'System': system,
            'Subsystem': subsystem,
            'Function': function
        }))
        count += 1

    if 'antibiotic' in func_lower and 'biosynthesis' in func_lower:
        add_kw('Secondary Metabolism', 'Bacterial cytostatics, differentiation factors and antibiotics',
               'Antibiotics biosynthesis')

    if 'dna polymerase' in func_lower:
        add_kw('DNA Metabolism', 'DNA replication', 'DNA polymerase')

    if 'methylase' in func_lower and 'rna' not in func_lower and ('dna' in func_lower or 'modification' in func_lower):
        add_kw('DNA Metabolism', 'DNA Metabolism - no subcategory', 'DNA methylation')

    if re.match(r'^T[1-6]SS', function):
        romanNumbers = ['I', 'II', 'III', 'IV', 'V', 'VI']
        type_num = re.search(r'[1-6]', function).group(0)
        add_kw('Membrane Transport', f'Protein secretion system, Type {romanNumbers[int(type_num) - 1]}',
               f'T{type_num} SS component')

    if 'integrase' in func_lower:
        add_kw('Phages, Prophages, Transposable elements, Plasmids', 'Transposable elements', 'Integrase')

    if 'transposon' in func_lower:
        add_kw('Phages, Prophages, Transposable elements, Plasmids', 'Transposable elements', 'Transposon')

    if 'transposase' in func_lower:
        add_kw('Phages, Prophages, Transposable elements, Plasmids', 'Transposable elements', 'Transposase')

    if 'response regulator' in func_lower or 'histidine kinase' in func_lower or (
            'two-component' in func_lower and ('response' in func_lower or 'sensor' in func_lower)):
        add_kw('Regulation and Cell signaling', 'Regulation and Cell signaling - no subcategory',
               'Two-component regulatory system')
        if 'dna' in func_lower and ' binding' in func_lower:
            add_kw('RNA Metabolism', 'Transcription', 'Two-component regulatory system')

    if 'nuclease' in func_lower:
        if 'dna' in func_lower or 'deoxyribo' in func_lower:
            add_kw('DNA Metabolism', 'DNA Metabolism - no subcategory', 'Nuclease')
        if ('rna' in func_lower or 'ribo' in func_lower) and 'deoxyribo' not in func_lower:
            add_kw('RNA Metabolism', 'RNA processing and modification', 'Nuclease')
        if 'dna' not in func_lower and 'rna' not in func_lower and 'ribo' not in func_lower:
            add_kw('DNA Metabolism', 'DNA Metabolism - no subcategory', 'Nuclease')
            add_kw('RNA Metabolism', 'RNA processing and modification', 'Nuclease')

    if 'siderophore' in func_lower:
        add_kw('Membrane Transport', 'Membrane Transport - no subcategory', 'Siderophore')

    if 'rod shape' in func_lower:
        add_kw('Cell Wall and Capsule', 'Cell Wall and Capsule - no subcategory', 'Cell shape')

    if 'protein translocase' in func_lower:
        add_kw('Membrane Transport', 'Protein transport', 'Protein transport')

    if 'dipeptidase' in func_lower:
        add_kw('Protein Metabolism', 'Protein degradation', 'Dipeptidases')

    if (
            'protease' in func_lower or 'peptidase' in func_lower) and 'aminopeptidase' not in func_lower and 'dipeptidase' not in func_lower and 'synthase' not in func_lower:
        add_kw('Protein Metabolism', 'Protein degradation', 'Protein degradation')

    if 'sigma factor' in func_lower:
        add_kw('RNA Metabolism', 'Transcription', 'Transcription initiation, bacterial sigma factors')

    if 'rna' in func_lower and ('methyltransferase' in func_lower or 'mnm' in func_lower or 'methylase' in func_lower):
        add_kw('RNA Metabolism', 'RNA processing and modification', 'RNA methylation')

    if 'rna' in func_lower and 'methylthiotransferase' in func_lower:
        add_kw('RNA Metabolism', 'RNA processing and modification', 'tRNA methylthiolation')

    if ('polyribonucleotide' in func_lower or 'rna' in func_lower) and 'nucleotidyltransferase' in func_lower:
        add_kw('RNA Metabolism', 'RNA processing and modification', 'RNA processing and degradation, bacterial')

    if 'rna' in func_lower and 'pseudouridine' in func_lower:
        add_kw('RNA Metabolism', 'RNA processing and modification', 'Pseudouridinylation')

    if 'aminopeptidase' in func_lower:
        add_kw('Protein Metabolism', 'Protein degradation', 'Aminopeptidases')

    if 'helicase' in func_lower:
        if 'dna' in func_lower:
            add_kw('DNA Metabolism', 'DNA Metabolism - no subcategory', 'DNA helicase')
        if 'rna' in func_lower:
            add_kw('RNA Metabolism', 'RNA Metabolism - no subcategory', 'RNA helicase')

    if 'aerotolerance' in func_lower or 'BatA' in function or 'BatB' in function or 'BatC' in function or 'BatD' in function or 'BatE' in function:
        add_kw('Stress Response', 'Oxidative stress', 'Aerotolerance')

    if 'vgrg' in func_lower:
        add_kw('Membrane Transport', 'Protein secretion system, Type VI', 'Actin cross-linking toxin')

    if 'cell division' in func_lower and ('protein' in func_lower or 'trigger' in func_lower):
        add_kw('Cell Division and Cell Cycle', 'Cell Division and Cell Cycle - no subcategory', 'Cell division protein')

    if 'restriction' in func_lower and ('modification' in func_lower or 'methylase' in func_lower):
        if 'type i ' in func_lower:
            add_kw('DNA Metabolism', 'DNA Metabolism - no subcategory', 'Type I Restriction-Modification')
        if 'type ii ' in func_lower:
            add_kw('DNA Metabolism', 'DNA Metabolism - no subcategory', 'Type II Restriction-Modification')
        if 'type iii ' in func_lower:
            add_kw('DNA Metabolism', 'DNA Metabolism - no subcategory', 'Type III Restriction-Modification')
        if 'type iv ' in func_lower:
            add_kw('DNA Metabolism', 'DNA Metabolism - no subcategory', 'Type IV Restriction-Modification')

    if 'restriction enzyme' in func_lower or 'restriction endonuclease' in func_lower:
        add_kw('DNA Metabolism', 'DNA Metabolism - no subcategory', 'Restriction Enzyme')

    if 'cluster' in func_lower and 'iron' in func_lower and 'sulfur' in func_lower:
        add_kw('Miscellaneous', 'Plant-Prokaryote DOE project', 'Iron-sulfur cluster assembly')

    if 'lipoprotein' in func_lower and 'releas' in func_lower:
        add_kw('Membrane Transport', 'ABC transporters', 'Lipoprotein-releasing system')

    if 'lipopolysaccharide' in func_lower and 'synthesis' in func_lower and 'export' not in func_lower and 'capsular' not in func_lower:
        add_kw('Cell Wall and Capsule', 'Gram-Negative cell wall components', 'Lipopolysaccharides')

    if 'polysaccharide' in func_lower and 'synthesis' in func_lower and 'export' not in func_lower and 'capsular' not in func_lower:
        add_kw('Carbohydrates', 'Polysaccharides', 'Polysaccharide biosynthesis')

    if 'lpt' in func_lower and 'protein' in func_lower:
        add_kw('Cell Wall and Capsule', 'Gram-Negative cell wall components', 'Lipoprotein sorting system')

    if 'capsular' in func_lower and 'polysaccharide' in func_lower and 'synthesis' in func_lower:
        add_kw('Cell Wall and Capsule', 'Capsular and extracellular polysacchrides',
               'Capsular Polysaccharides Biosynthesis and Assembly')

    if 'crisp' in func_lower and 'repeat' in func_lower:
        add_kw('DNA Metabolism', 'CRISPs', 'CRISPRs')

    if 'crisp' in func_lower and 'spacer' in func_lower:
        add_kw('DNA Metabolism', 'CRISPs', 'Spacers')

    if 'crisp' in func_lower and ('ramp' in func_lower or 'cas' in func_lower):
        add_kw('DNA Metabolism', 'CRISPs', 'CRISPR-associated proteins')

    if 'transport' in func_lower and 'ABC' not in function and (
            'antiport' not in func_lower or 'symport' not in func_lower or 'uniport' not in func_lower):
        add_kw('Membrane Transport', 'Membrane Transport - no subcategory', 'none')

    if ('ABC' not in function and 'antiport' in func_lower) or 'symport' in func_lower or 'uniport' in func_lower:
        add_kw('Membrane Transport', 'Uni- Sym- and Antiporters', 'none')

    if 'resist' in func_lower and 'phage' not in func_lower:
        add_kw('Virulence, Disease and Defense', 'Resistance to antibiotics and toxic compounds', 'none')

    if 'resist' in func_lower and 'phage' in func_lower:
        add_kw('Virulence, Disease and Defense', 'Virulence, Disease and Defense - no subcategory', 'Phage defence')

    if 'cytochrome' in func_lower:
        add_kw('Respiration', 'Respiration - no subcategory', 'Cytochrome')

    if 'heat' in func_lower and 'shock' in func_lower:
        add_kw('Stress Response', 'Heat shock', 'none')

    if 'cold' in func_lower and 'shock' in func_lower:
        add_kw('Stress Response', 'Cold shock', 'none')
        add_kw('Stress Response', 'Stress Response - no subcategory', 'Phage shock')

    if 'ABC' in function and 'transport' in func_lower:
        add_kw('Membrane Transport', 'ABC-transporters', 'none')

    if 'tonB' in function or 'TonB' in function:
        add_kw('Membrane Transport', 'Membrane Transport - no subcategory', 'Ton and Tol transopt system')

    if 'ribosomal protein' in func_lower:
        add_kw('Protein Metabolism', 'Protein biosynthesis', 'Ribosome LSU/SSU bacterial')

    if 'dna' in func_lower and 'repair' in func_lower:
        add_kw('DNA Metabolism', 'DNA repair', 'none')

    if 'tRNA-' in function and len(function) < 16:
        add_kw('RNA Metabolism', 'RNA processing and modification', 'none')

    if 'transcription' in func_lower and 'regulator' in func_lower:
        add_kw('RNA Metabolism', 'RNA processing and modification', 'none')

    if 'type' in func_lower and 'secretion' in func_lower:
        add_kw('Membrane Transport', 'Protein secretion system', 'none')

    if 'topoisomerase' in func_lower:
        add_kw('DNA Metabolism', 'DNA replication', 'DNA topoisomerases')

    if 'replication' in func_lower:
        add_kw('DNA Metabolism', 'DNA replication', 'none')

    if count > 0:
        data.kwCls_functions_set.add(function)

    data.processed_kw_functions.add(function)
    return count

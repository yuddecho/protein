# finally 裁剪长度
files = ['tt_pdb_solu', 'tt_insolu', 'chang_solu', 'chang_insolu', 'nesg_solu', 'nesg_insolu']
finally_number = {}
# seq_min, seq_max = 30, 622
seq_min, seq_max = 20, 1186
seq_count = 0
root = '../../data'

for fasta_file_name in files:
    fasta_file = f'{root}/db/tmbed/{fasta_file_name}.fasta'
    target_file = f'{root}/db/finally/{fasta_file_name}.fasta'

    with open(target_file, 'w', encoding='utf-8') as w:
        with open(fasta_file, 'r', encoding='utf-8') as r:
            while True:
                fasta = r.readline()
                seq = r.readline()

                seq = seq.strip()

                if not seq:
                    finally_number[fasta_file_name] = seq_count
                    seq_count = 0
                    break

                # 限制长度
                seq_len = len(seq)
                if seq_len < seq_min or seq_len > seq_max:
                    continue

                seq_count += 1

                w.write(fasta)
                w.write(f'{seq}\n')

print(finally_number)

"""
1. 分割文件最大能超过30000
2. 生成 tmbed 运行命令
3. 处理 tmbed 运行结果
4. 合并文件

在服务器中运行：
    ../../data 对应为 ~/protein
    ./db/tmbed 对应 ~/protein/tmbed

"""
# 文件路径设置 (~/protein)
root = '../../data'

# (~/protein)
web_root = f'{root}/db/mmseqs'

train_set_files = ['tt_pdb_solu', 'tt_insolu']
test_set_files = ['chang_solu', 'chang_insolu', 'nesg_solu', 'nesg_insolu']

# 脚本文件
sh_file = f'{root}/tmbed.sh'
sh_w = open(sh_file, 'w', encoding='utf-8')

# 处理 test_set, 先 embed 再 predict
for tag in range(4):
    sh_w.write(
        f'CUDA_VISIBLE_DEVICES={tag} nohup python -m tmbed embed --batch-size 500 -f ~/protein/res/{test_set_files[tag]}.fasta -e ~/protein/{test_set_files[tag]}_embed.h5 > ~/protein/{test_set_files[tag]}_embed.log 2>&1 &\n')

sh_w.write(f'wait\n')

for tag in range(4):
    sh_w.write(
        f'CUDA_VISIBLE_DEVICES={tag} nohup python -m tmbed predict --batch-size 1000 -f ~/protein/res/{test_set_files[tag]}.fasta -e ~/protein/{test_set_files[tag]}_embed.h5 -p ~/protein/{test_set_files[tag]}.pred > ~/protein/{test_set_files[tag]}_pred.log 2>&1 &\n')

sh_w.write(f'wait\n')


# 处理 train_set: 大文件分批预测
def train_set_script(_w, file_root_path, file_name, tag_max, nums_max):
    file_path = f'{file_root_path}/res/{file_name}.fasta'
    count = 0

    # for _tag in range(tag_max):
    _tag = 0
    total = 0
    with open(file_path, 'r', encoding='utf-8') as r:
        target_file = f'{file_root_path}/{file_name}_{_tag}.fasta'
        target_w = open(target_file, 'w', encoding='utf-8')

        while True:
            fasta = r.readline()
            seq = r.readline()

            # 文件结束
            seq = seq.strip()
            if not seq:
                print(f'{target_file}: count {count}')
                print(f'{file_path}: total {total}')
                target_w.close()
                break

            # 统计一个有效数据
            count += 1
            target_w.write(fasta)
            target_w.write(f'{seq}\n')

            total += 1

            # 一个子文件的最大数量
            if count == nums_max:
                print(f'{target_file}: count {count}')
                target_w.close()

                # 更新变量
                count = 0
                _tag += 1
                target_file = f'{file_root_path}/{file_name}_{_tag}.fasta'
                target_w = open(target_file, 'w', encoding='utf-8')

        for __tag in range(tag_max):
            if __tag != 0 and __tag % 4 == 0:
                _w.write(f'wait\n')

            _w.write(f'CUDA_VISIBLE_DEVICES={__tag % 4} nohup python -m tmbed embed --batch-size 500 -f ~/protein/res/{file_name}_{__tag}.fasta -e ~/protein/{file_name}_{__tag}.h5 > ~/protein/{file_name}_{__tag}_embed.log 2>&1 &\n')

        _w.write(f'wait\n')

        for __tag in range(tag_max):
            if __tag != 0 and __tag % 4 == 0:
                _w.write(f'wait\n')

            sh_w.write(f'CUDA_VISIBLE_DEVICES={__tag % 4} nohup python -m tmbed predict --batch-size 1000 -f ~/protein/res/{file_name}_{__tag}.fasta -e ~/protein/{file_name}_{__tag}.h5 -p ~/protein/{file_name}_{__tag}.pred > ~/protein/{file_name}_{__tag}_pred.log 2>&1 &\n')

        sh_w.write(f'wait\n')


tt_pdb_solu = train_set_files[0]
tt_insolu = train_set_files[1]

# tt_pdb_solu 一共 67647
max_tag, nums = 68, 1000
train_set_script(sh_w, web_root, tt_pdb_solu, max_tag, nums)

# tt_insolu : 125979
max_tag, nums = 126, 1000
train_set_script(sh_w, web_root, tt_insolu, max_tag, nums)

# 处理结果并合并文件
sh_w.write('mkdir ~/protein/tmbed\n')

sh_w.write(
    """cat <<EOF > res.py
# 处理结果函数 .pred -> .fasta
def get_res(source_file, res_file):
    with open(res_file, 'w', encoding='utf-8') as w:
        with open(source_file, 'r', encoding='utf-8') as r:
            while True:
                fasta = r.readline()
                seq = r.readline()
                result = r.readline()
                
                result = result.strip()
                if not result:
                    break
                
                result_set = set(result)
                if len(result_set) != 1 or '.' not in result_set:
                    continue
                
                w.write(fasta)
                w.write(seq)


root = '~/protein'
tmbed_dir = f'{root}/tmbed'

train_set_files = ['tt_pdb_solu', 'tt_insolu']
test_set_files = ['chang_solu', 'chang_insolu', 'nesg_solu', 'nesg_insolu']

# 处理测试集 test set
for test_set in test_set_files:
    source = f'{root}/{test_set}.pred'
    target = f'{tmbed_dir}/{test_set}.fasta'
    get_res(source, target)

# 处理训练集 train set
for train_set, max_tag in zip(train_set_files, [68, 126]):
    for tag in range(max_tag):
        source = f'{root}/{train_set}_{tag}.pred'
        target = f'{root}/{train_set}_{tag}.fasta'
        get_res(source, target)

# 合并文件
for train_set, max_tag in zip(train_set_files, [68, 126]):
    # 合并结果
    target = f'{tmbed_dir}/{train_set}.fasta'
    with open(target, 'w', encoding='utf-8') as w:
        # 逐一打开子文件
        for tag in range(max_tag):
            source = f'{root}/{train_set}_{tag}.fasta'
            with open(source, 'r', encoding='utf-8') as r:
                while True:
                    fasta = r.readline()
                    seq = r.readline()
    
                    fasta = fasta.strip()
                    if not fasta:
                        break
    
                    w.write(f'{fasta}\\n')
                    w.write(seq)

EOF

python res.py

wait

"""
)


sh_w.close()

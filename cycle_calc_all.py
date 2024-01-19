import statistics
import os

from cycle_calc import cycle_calc

def test_cycle_calc_all():    
    '''
    自动测试所有音频的预测情况。
    '''
    data_folder = 'data/icbhi_dataset/audio_test_data'
    
    fileset = set()
    for filename in os.listdir(data_folder):
        f = filename.split('.')[0]
        fileset.add(f)
    
    succ_cnt, fail_cnt, succ_rate = [0, 0], [0, 0], [0, 0]
    error_sum = 0
    
    for f in fileset:
        pred_list, pred_freq, label_list, label_freq = cycle_calc(os.path.join(data_folder, f), False)
        # 枚举标签中的每一个呼吸周期，看看能不能从预测序列中找到一个匹配上。

        cur_succ_cnt = [0, 0]
        cur_fail_cnt = [0, 0]
        
        for i in range(len(label_list) - 1):
            success = [False, False]
            for j in range(len(pred_list) - 1):
                error = abs(label_list[i] - pred_list[j])
                if (error < 0.3):
                    success[0] = True
                    error_sum += error
                    if (abs(label_list[i + 1] - pred_list[j + 1]) < 0.3):
                        success[1] = True
                    break
            
            for i in range(2):
                if success[i]:
                    cur_succ_cnt[i] += 1
                else:
                    cur_fail_cnt[i] += 1
                
        # if current_succ / (current_succ + current_fail) < 0.5:
        #     ...

        for i in range(2):
            succ_cnt[i] += cur_succ_cnt[i]
            fail_cnt[i] += cur_fail_cnt[i]

        print('(success rate){}: {:.3f}, {:.3f}'.format(f, 
                                                        cur_succ_cnt[0] / (cur_succ_cnt[0] + cur_fail_cnt[0]),
                                                        cur_succ_cnt[1] / (cur_succ_cnt[1] + cur_fail_cnt[1])))
    
    for i in range(2):
        succ_rate[i] = succ_cnt[i] / (succ_cnt[i] + fail_cnt[i])

    average_error = error_sum / succ_cnt[0]

    print('succes rate: {:.3f}, {:.3f}'.format(succ_rate[0], succ_rate[1]))
    print('average error: {}'.format(average_error))
    
    # return succ_rate, average_error            
    
test_cycle_calc_all()
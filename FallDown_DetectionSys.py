
# ************************************ IMPORT LIBRARY ********************************************
import cv2
from yolov7_pose.detect_pose import Y7Detect, draw_kpts, draw_boxes
import time
import numpy as np
from numpy import random
from strong_sort.strong_sort import StrongSORT
from pathlib import Path
import torch
import argparse
from classification_lstm.utils.load_model import Model
from classification_stgcn.Actionsrecognition.ActionsEstLoader import TSSTG
import random
import requests
from qiniu import Auth, put_file, etag, urlsafe_base64_encode, CdnManager
import os
# *********************************** CONFIG qiniu  *************************
# 需要填写你的 Access Key 和 Secret Key
access_key = 'oehKGwSym_EJ03gQctqa9UfL_5JsFHN3Q4qqZAi6'
secret_key = 'eVGPNFjlsiVfRfVxFx7gtNBP3QPBmJufU3xfm_VJ'
# 构建鉴权对象
q = Auth(access_key, secret_key)

# 初始化 Auth 和 CdnManager
q = Auth(access_key, secret_key)
cdn_manager = CdnManager(q)

# 待刷新的 URL 列表，可以传入多个 URL
urls = [
    'http://ru6hajslm.hn-bkt.clouddn.com/fall_down_img.png',
]

# 要上传的空间名称
bucket_name = 'fall-down'
# 上传到七牛后保存的文件名
key = 'fall_down_img.png'


# *********************************** 配置路径并重置CUDA *************************

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
WEIGTHS = ROOT

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
print(torch.cuda.is_available())


# *********************************** 处理 和 运行 *********************************



fall_down_status = 0


def information_push():
    global fall_down_status
    """
    function: when people fall-down,WeChat information push
    """
    if fall_down_status == 20:

        # 上传跌倒图片
        #Begin
        localfile = 'fall_down_img.png'
        # 生成上传 Token，可以指定过期时间等参数
        token = q.upload_token(bucket_name, key, 3600)
        # 调用 put_file 方法上传图片
        ret, info = put_file(token, key, localfile)
        if info.status_code == 200:
            print('上传成功，返回的信息为：', ret)
        else:
            print('上传失败，错误信息为：', info.error)
        #END

        # 刷新指定 URL 的 CDN 缓存
        refresh_result = cdn_manager.refresh_urls(urls)
        # 输出code：200表示刷新成功
        #print(refresh_result)

        time.sleep(2.5)
        # 请求之前再次刷新外链，确保图片更新完成
        refresh_result = cdn_manager.refresh_urls(urls)

        # 获取公共的外链
        base_url = 'http://%s/%s' % ('ru6hajslm.hn-bkt.clouddn.com', key)
        public_url = q.private_download_url(base_url, expires=3600)
        # public_url 为公共外链
        print(public_url)

        time_tuple = time.localtime(time.time())
        t1 = "{}年{}月{}日{}点".format(time_tuple[0], time_tuple[1], time_tuple[2], time_tuple[3])
        t2 = "北京时间{}年{}月{}日{}点{}分{}秒".format(time_tuple[0], time_tuple[1], time_tuple[2], time_tuple[3],time_tuple[4], time_tuple[5])

        refresh_result = cdn_manager.refresh_urls(urls)

        response = requests.get('http://www.pushplus.plus/send?token=1e87ee03f404402c9f8264a73125972c&title={}跌倒提醒&content=在{}通过摄像头检测有人跌倒，请查看图片确认跌倒情况。<br> 如情况属实请尽快处理！ <br><img src="{}" />"&template=html'.format(t1, t2, public_url))

        if response.status_code == 200:
            print(response.text)
        else:
            print('Error: ', response.status_code)


def detect_video(url_video=None, flag_save=False, fps=None, name_video='video.avi'):
    # ******************************** 加载模型 *************************************************
    # 加载模型检测 yolov7 姿势
    global fall_down_status
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    y7_pose = Y7Detect()
    class_name = y7_pose.class_names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_name]

    # *************************** 加载模型LSTM或ST-GCN ************************************************
    # LSTM
    # action_model = Model(device=device, skip=True)

    # ST-GCN 时间积卷 动作识别
    action_model = TSSTG(device=device, skip=True)

    # **************************** 初始化跟踪 *************************************************
    tracker = StrongSORT(device=device, max_age=30, n_init=3, max_iou_distance=0.7)  # deep sort

    # ********************************** 读取视频 **********************************************
    if url_video == '':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(url_video)

    # 获取大小
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    print(frame_height, frame_width)
    h_norm, w_norm = 720, 1280
    if frame_height > h_norm and frame_width > w_norm:
        frame_width = w_norm
        frame_height = h_norm
    # 获取相机的fps
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)


    # ******************************** 实时监测 ********************************************
    memory = {}  # 记忆包含识别人类行为
    count = True  # 跳过帧
    turn_detect_face = False  # Ture:关闭人脸识别
    while True:
        start = time.time()
        # ************************************ 获取帧 *************************************
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape

        if h > h_norm or w > w_norm:
            rate_max = max(h_norm / h, w_norm / w)
            frame = cv2.resize(frame, (int(rate_max * w), int(rate_max * h)), interpolation=cv2.INTER_AREA)
            h, w, _ = frame.shape
        # ************************************* 姿势检测 ***********************************
        if count:
            bbox, label, score, label_id, kpts = y7_pose.predict(frame)
            id_hold = []
            for i, box in enumerate(bbox):
                # check and remove bbox
                if box[0] < 10 or box[1] < 10 or box[2] > w - 10 or box[3] > h - 10:
                    id_hold.append(False)
                    continue
                id_hold.append(True)
            bbox, score, kpts = np.array(bbox), np.array(score), np.array(kpts)
            bbox, score, kpts = bbox[id_hold], score[id_hold], kpts[id_hold]


        # ************************************* 追踪 *******************************************
        if len(bbox) != 0:
            if count:
                data = tracker.update(bbox, score, kpts, frame)
            for outputs in data:
                if len(outputs['bbox']) != 0:
                    box, kpt, track_id, list_kpt = outputs['bbox'], outputs['kpt'], outputs['id'], \
                        outputs['list_kpt']
                    kpt = kpt[:, :2].astype('int')

                    icolor = class_name.index('0')
                    draw_boxes(frame, box, color=colors[icolor])
                    draw_kpts(frame, [kpt])
                    color = (0, 255, 255)
                    color1 = (255, 255, 0)

                    # ************************************ 行为预测 ********************************
                    if len(list_kpt) == 15:
                        # action, score = action_model.predict([list_kpt], w, h, batch_size=1)
                        action, score = action_model.predict(list_kpt, (w, h))
                    try:
                        if action[0] == "Fall Down":
                            color = (0, 0, 255)
                            # 使用 CV2 保存当前跌倒图片
                            cv2.imwrite('fall_down_img.png', frame)
                            time.sleep(0.1)  # 暂停 0.1 秒钟
                            fall_down_status = fall_down_status + 1
                            information_push()
                        else:
                            fall_down_status = 0

                        cv2.putText(frame, '{}'.format(action[0]),
                                    (max(box[0] - 20, 0), box[1] + 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
                    except:

                        cv2.putText(frame, '{}'.format('Pending ...'),
                                    (max(box[0] - 20, 0), box[1] + 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

            # 用于遍历键值，如果某个键对应的计数器memory[key][1]大于30，就删除这个键，否则将计数器加1，更新字典。
            # 如果某个键对应的计数器memory[key][1]大于30，就删除这个键，否则将计数器加1，更新字典。
            keys = list(memory.keys())
            for key in keys:
                if memory[key][1] > 30:
                    del memory[key]
                    continue
                memory.update({key: [memory[key][0], memory[key][1] + 1]})

        # ******************************************** 跳帧 *******************************************
        count = not count
        # ******************************************** 显示 *******************************************
        fps = int(1 / (time.time() - start))
        cv2.putText(frame, 'FPS:' + str(fps), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('video', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect Face On Video')
    parser.add_argument("-fn", "--file_name", help="video file name or rtsp", default='', type=str)
    parser.add_argument("-op", "--option", help="if save video then choice option is True or False", default=True,type=bool)
    parser.add_argument("-o", "--output", help="path to output video file", default='recog_recording.avi', type=str)
    parser.add_argument("-f", "--fps", default=30, help="FPS of output video", type=int)
    args = parser.parse_args()
    source = args.file_name
    cv2.namedWindow('video')
    # if run  as terminal, replace url = source
    detect_video(url_video=source, flag_save=args.option, fps=args.fps, name_video=args.output)

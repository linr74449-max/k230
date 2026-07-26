from libs.PipeLine import PipeLine, ScopedTiming
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
import os, gc
from media.media import *
from time import *
import nncase_runtime as nn
import ulab.numpy as np
import image
import aidemo


class YOLOv12App(AIBase):

    def __init__(self,
                 kmodel_path,
                 model_input_size,
                 anchors,
                 confidence_threshold=0.50,
                 nms_threshold=0.35,
                 rgb888p_size=[1280,720],
                 display_size=[800,480],
                 debug_mode=0):

        super().__init__(
            kmodel_path,
            model_input_size,
            rgb888p_size,
            debug_mode
        )

        self.class_id=["steel_ball"]

        self.kmodel_path=kmodel_path
        self.model_input_size=model_input_size
        self.confidence_threshold=confidence_threshold
        self.detect_threshold=0.35
        self.nms_threshold=nms_threshold
        self.anchors=anchors

        self.rgb888p_size=[
            ALIGN_UP(rgb888p_size[0],16),
            rgb888p_size[1]
        ]

        self.display_size=[
            ALIGN_UP(display_size[0],16),
            display_size[1]
        ]

        self.debug_mode=debug_mode

        self.ai2d=Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(
            nn.ai2d_format.NCHW_FMT,
            nn.ai2d_format.NCHW_FMT,
            np.uint8,
            np.uint8
        )

        # 短时跟踪：晃动时允许连续几帧漏检，不立刻清掉目标框
        self.last_dets=[]
        self.miss_count=0
        self.max_miss_count=4
        self.smooth_alpha=0.65

        self.scale_x=self.rgb888p_size[0]/self.model_input_size[0]
        self.scale_y=self.scale_x


    def config_preprocess(self,input_image_size=None):

        with ScopedTiming(
            "set preprocess config",
            self.debug_mode>0
        ):
            ai2d_input_size = (
                input_image_size
                if input_image_size
                else self.rgb888p_size
            )

            top,bottom,left,right=self.get_padding_param()

            if self.debug_mode>0:
                print(
                    "padding:{} {} {} {}".format(
                        top,bottom,left,right
                    )
                )

            self.ai2d.pad(
                [
                    0,0,0,0,
                    top,bottom,left,right
                ],
                0,
                [104,117,123]
            )

            self.ai2d.resize(
                nn.interp_method.tf_bilinear,
                nn.interp_mode.half_pixel
            )

            self.ai2d.build(
                [
                    1,
                    3,
                    ai2d_input_size[1],
                    ai2d_input_size[0]
                ],
                [
                    1,
                    3,
                    self.model_input_size[1],
                    self.model_input_size[0]
                ]
            )


    def postprocess(self,results):

        det_res=[]

        with ScopedTiming(
            "postprocess",
            self.debug_mode>0
        ):
            output=results[0][0]

            for i in range(2100):
                result=output[:,i]
                max_score=max(result[4:])

                if max_score > self.detect_threshold:
                    x=result[0]*self.scale_x
                    y=result[1]*self.scale_y
                    w=result[2]*self.scale_x
                    h=result[3]*self.scale_y
                    cls_id=list(result[4:]).index(max_score)

                    det_res.append(
                        [
                            x,
                            y,
                            w,
                            h,
                            cls_id,
                            max_score
                        ]
                    )

            det_res.sort(
                key=lambda x:x[-1],
                reverse=True
            )

            det_res=self.nms(det_res,self.nms_threshold)

            show_res=[]
            for det in det_res:
                if det[-1]>=self.confidence_threshold:
                    show_res.append(det)

            if len(show_res)==0 and len(det_res)>0:
                show_res.append(det_res[0])

            show_res=self.temporal_filter(show_res)

        return show_res


    def nms(self,dets,iou_threshold):

        keep=[]

        while len(dets)>0:
            best=dets[0]
            keep.append(best)

            remain=[]
            for i in range(1,len(dets)):
                if self.box_iou(best,dets[i])<iou_threshold:
                    remain.append(dets[i])

            dets=remain

        return keep


    def box_iou(self,a,b):

        ax1=a[0]-a[2]/2
        ay1=a[1]-a[3]/2
        ax2=a[0]+a[2]/2
        ay2=a[1]+a[3]/2

        bx1=b[0]-b[2]/2
        by1=b[1]-b[3]/2
        bx2=b[0]+b[2]/2
        by2=b[1]+b[3]/2

        inter_x1=max(ax1,bx1)
        inter_y1=max(ay1,by1)
        inter_x2=min(ax2,bx2)
        inter_y2=min(ay2,by2)

        inter_w=max(0,inter_x2-inter_x1)
        inter_h=max(0,inter_y2-inter_y1)
        inter_area=inter_w*inter_h

        area_a=max(0,ax2-ax1)*max(0,ay2-ay1)
        area_b=max(0,bx2-bx1)*max(0,by2-by1)

        union_area=area_a+area_b-inter_area

        if union_area<=0:
            return 0

        return inter_area/union_area


    def center_dist(self,a,b):

        dx=a[0]-b[0]
        dy=a[1]-b[1]

        return dx*dx+dy*dy


    def temporal_filter(self,dets):

        if len(dets)>0:
            filtered=[]

            for det in dets:
                best_prev=None
                best_iou=0

                for prev in self.last_dets:
                    iou=self.box_iou(det,prev)
                    if iou>best_iou:
                        best_iou=iou
                        best_prev=prev

                if best_prev is not None:
                    dist2=self.center_dist(det,best_prev)

                    if best_iou>0.05 or dist2<120*120:
                        det[0]=self.smooth_alpha*det[0]+(1-self.smooth_alpha)*best_prev[0]
                        det[1]=self.smooth_alpha*det[1]+(1-self.smooth_alpha)*best_prev[1]
                        det[2]=self.smooth_alpha*det[2]+(1-self.smooth_alpha)*best_prev[2]
                        det[3]=self.smooth_alpha*det[3]+(1-self.smooth_alpha)*best_prev[3]

                filtered.append(det)

            self.last_dets=filtered
            self.miss_count=0

            return filtered

        if len(self.last_dets)>0 and self.miss_count<self.max_miss_count:
            self.miss_count+=1

            hold_dets=[]
            for det in self.last_dets:
                hold_det=[
                    det[0],
                    det[1],
                    det[2],
                    det[3],
                    det[4],
                    det[5]*0.85
                ]
                hold_dets.append(hold_det)

            self.last_dets=hold_dets
            return hold_dets

        self.last_dets=[]
        self.miss_count=0

        return []


    def draw_result(self,pl,dets):

        with ScopedTiming(
            "display_draw",
            self.debug_mode>0
        ):
            pl.osd_img.clear()
            pl.osd_img.draw_string_advanced(
                0,
                0,
                32,
                "balls: {}".format(len(dets)),
                color=(255,0,255,0)
            )

            if dets:
                for det in dets:
                    x,y,w,h=map(
                        lambda x:int(round(x,0)),
                        det[:4]
                    )

                    x=x*self.display_size[0]//self.rgb888p_size[0]
                    y=y*self.display_size[1]//self.rgb888p_size[1]
                    w=w*self.display_size[0]//self.rgb888p_size[0]
                    h=h*self.display_size[1]//self.rgb888p_size[1]

                    x1=x-w//2
                    y1=y-h//2

                    if x1<0:
                        x1=0

                    if y1<0:
                        y1=0

                    if x1+w>self.display_size[0]:
                        w=self.display_size[0]-x1

                    if y1+h>self.display_size[1]:
                        h=self.display_size[1]-y1

                    if w<=0 or h<=0:
                        continue

                    pl.osd_img.draw_rectangle(
                        x1,
                        y1,
                        w,
                        h,
                        color=(255,0,255,0),
                        thickness=2
                    )

                    pl.osd_img.draw_string_advanced(
                        x1,
                        y1,
                        32,
                        "{} {}".format(
                            self.class_id[det[-2]],
                            round(det[-1],2)
                        ),
                        color=(255,0,255,0)
                    )


    def get_padding_param(self):

        dst_w=self.model_input_size[0]
        dst_h=self.model_input_size[1]

        ratio_w=dst_w/self.rgb888p_size[0]
        ratio_h=dst_h/self.rgb888p_size[1]

        ratio=min(ratio_w,ratio_h)

        new_w=int(
            ratio*self.rgb888p_size[0]
        )

        new_h=int(
            ratio*self.rgb888p_size[1]
        )

        dw=(dst_w-new_w)/2
        dh=(dst_h-new_h)/2

        top=0
        bottom=int(round(dh*2+0.1))
        left=0
        right=int(round(dw*2-0.1))

        return top,bottom,left,right


if __name__=="__main__":

    display_mode="lcd"

    rgb888p_size=[1280,720]
    display_size=[800,480]

    kmodel_path="/sdcard/best.kmodel"

    confidence_threshold=0.50
    nms_threshold=0.35

    anchors=None

    pl=PipeLine(
        rgb888p_size=rgb888p_size,
        display_size=display_size,
        display_mode=display_mode
    )

    pl.create()

    yolo_det=YOLOv12App(
        kmodel_path,
        model_input_size=[320,320],
        anchors=anchors,
        confidence_threshold=confidence_threshold,
        nms_threshold=nms_threshold,
        rgb888p_size=rgb888p_size,
        display_size=display_size,
        debug_mode=0
    )

    yolo_det.config_preprocess()

    frame_count=0

    try:
        while True:
            frame_count+=1
            os.exitpoint()

            with ScopedTiming("total",0):
                img=pl.get_frame()
                res=yolo_det.run(img)
                yolo_det.draw_result(pl,res)
                pl.show_image()

                if frame_count%60==0:
                    gc.collect()

    except Exception as e:
        print(e)

    finally:
        yolo_det.deinit()
        pl.destroy()

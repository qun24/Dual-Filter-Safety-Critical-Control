/*****************************************************************************
*	Implemetation of the saliency detction method described in paper
*	"Minimum Barrier Salient Object Detection at 80 FPS", Jianming Zhang, 
*	Stan Sclaroff, Zhe Lin, Xiaohui Shen, Brian Price, Radomir Mech, ICCV, 
*       2015
*	
*	Copyright (C) 2015 Jianming Zhang
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*
*	This program is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License
*	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
*	If you have problems about this software, please contact: 
*       jimmie33@gmail.com
*******************************************************************************/
#pragma once

#include "MBS.hpp"
#include <algorithm>



#define MAX_IMG_DIM 320 //越小消耗的资源越小
#define FRAME_MAX 30
#define SOBEL_THRESH 0.3
// SOBEL_THRESH 决定了哪些边缘被认为是“显著”的,值设得太低，会导致太多的噪声被认为是边缘；如果设得太高，则可能忽略一些实际存在的边缘
// 在噪声较多的图像中，可以增加 SOBEL_THRESH 的值以减少虚假边缘的影响

//初始化显著性图
MBS::MBS(const Mat& src)
    : mAttMapCount(0), mSrc(src) {
    mSaliencyMap = Mat::zeros(src.size(), CV_32FC1);
    split(mSrc, mFeatureMaps);//将输入图像 src 拆分成多个通道，存储在 mFeatureMaps 中
    for (auto& map : mFeatureMaps) {
        medianBlur(map, map,5);//对每个特征图（通道）应用中值滤波器进行平滑处理
    }
}

void MBS::computeSaliency(bool use_geodesic) {
    mMBSMap = use_geodesic ? fastGeodesic(mFeatureMaps) : fastMBS(mFeatureMaps);//根据 use_geodesic 参数，选择 fastGeodesic 或 fastMBS 方法计算显著性图
	// 使用固定阈值
	double fixedThreshold = 40; // 选择合适的固定阈值 warehouse
	threshold(mMBSMap, mMBSMap, fixedThreshold, 1.0, THRESH_TOZERO);
    // normalize(mMBSMap, mMBSMap, 0, 1.0, NORM_MINMAX);
    mSaliencyMap = mMBSMap;
}

Mat MBS::getSaliencyMap()
{
	Mat ret;
	normalize(mSaliencyMap, ret, 0.0, 255.0, NORM_MINMAX);//将显著性图 mSaliencyMap 归一化到 0 到 255 的范围
	ret.convertTo(ret, CV_8UC1);//转换为 8 位单通道图像
	return ret;
}


void rasterScan(const Mat& featMap, Mat& map, Mat& lb, Mat& ub)//进行正向栅格扫描，用于更新显著性图中的像素值。
{
	
	Size sz = featMap.size();
	float *pMapup = (float*)map.data + 1;
	float *pMap = pMapup + sz.width;
	uchar *pFeatup = featMap.data + 1;
	uchar *pFeat = pFeatup + sz.width;
	uchar *pLBup = lb.data + 1;
	uchar *pLB = pLBup + sz.width;
	uchar *pUBup = ub.data + 1;
	uchar *pUB = pUBup + sz.width;

	float mapPrev;
	float featPrev;
	uchar lbPrev, ubPrev;

	float lfV, upV;
	int flag;
	for (int r = 1; r < sz.height - 1; r++)
	{
		mapPrev = *(pMap - 1);
		featPrev = *(pFeat - 1);
		lbPrev = *(pLB - 1);
		ubPrev = *(pUB - 1);


		for (int c = 1; c < sz.width - 1; c++)
		{
			lfV = MAX(*pFeat, ubPrev) - MIN(*pFeat, lbPrev);//(*pFeat >= lbPrev && *pFeat <= ubPrev) ? mapPrev : mapPrev + abs((float)(*pFeat) - featPrev);
			upV = MAX(*pFeat, *pUBup) - MIN(*pFeat, *pLBup);//(*pFeat >= *pLBup && *pFeat <= *pUBup) ? *pMapup : *pMapup + abs((float)(*pFeat) - (float)(*pFeatup));

			flag = 0;
			if (lfV < *pMap)
			{
				*pMap = lfV;
				flag = 1;
			}
			if (upV < *pMap)
			{
				*pMap = upV;
				flag = 2;
			}

			switch (flag)
			{
			case 0:		// no update
				break;
			case 1:		// update from left
				*pLB = MIN(*pFeat, lbPrev);
				*pUB = MAX(*pFeat, ubPrev);
				break;
			case 2:		// update from up
				*pLB = MIN(*pFeat, *pLBup);
				*pUB = MAX(*pFeat, *pUBup);
				break;
			default:
				break;
			}

			mapPrev = *pMap;
			pMap++; pMapup++;
			featPrev = *pFeat;
			pFeat++; pFeatup++;
			lbPrev = *pLB;
			pLB++; pLBup++;
			ubPrev = *pUB;
			pUB++; pUBup++;
		}
		pMapup += 2; pMap += 2;
		pFeat += 2; pFeatup += 2;
		pLBup += 2; pLB += 2;
		pUBup += 2; pUB += 2;
	}
}


void invRasterScan(const Mat& featMap, Mat& map, Mat& lb, Mat& ub)//进行逆向栅格扫描，用于更新显著性图中的像素值
{
	
	Size sz = featMap.size();
	int datalen = sz.width*sz.height;
	float *pMapdn = (float*)map.data + datalen - 2;
	float *pMap = pMapdn - sz.width;
	uchar *pFeatdn = featMap.data + datalen - 2;
	uchar *pFeat = pFeatdn - sz.width;
	uchar *pLBdn = lb.data + datalen - 2;
	uchar *pLB = pLBdn - sz.width;
	uchar *pUBdn = ub.data + datalen - 2;
	uchar *pUB = pUBdn - sz.width;
	
	float mapPrev;
	float featPrev;
	uchar lbPrev, ubPrev;

	float rtV, dnV;
	int flag;
	for (int r = 1; r < sz.height - 1; r++)
	{
		mapPrev = *(pMap + 1);
		featPrev = *(pFeat + 1);
		lbPrev = *(pLB + 1);
		ubPrev = *(pUB + 1);

		for (int c = 1; c < sz.width - 1; c++)
		{
			rtV = MAX(*pFeat, ubPrev) - MIN(*pFeat, lbPrev);//(*pFeat >= lbPrev && *pFeat <= ubPrev) ? mapPrev : mapPrev + abs((float)(*pFeat) - featPrev);
			dnV = MAX(*pFeat, *pUBdn) - MIN(*pFeat, *pLBdn);//(*pFeat >= *pLBdn && *pFeat <= *pUBdn) ? *pMapdn : *pMapdn + abs((float)(*pFeat) - (float)(*pFeatdn));

			flag = 0;
			if (rtV < *pMap)
			{
				*pMap = rtV;
				flag = 1;
			}
			if (dnV < *pMap)
			{
				*pMap = dnV;
				flag = 2;
			}

			switch (flag)
			{
			case 0:		// no update
				break;
			case 1:		// update from right
				*pLB = MIN(*pFeat, lbPrev);
				*pUB = MAX(*pFeat, ubPrev);
				break;
			case 2:		// update from down
				*pLB = MIN(*pFeat, *pLBdn);
				*pUB = MAX(*pFeat, *pUBdn);
				break;
			default:
				break;
			}

			mapPrev = *pMap;
			pMap--; pMapdn--;
			featPrev = *pFeat;
			pFeat--; pFeatdn--;
			lbPrev = *pLB;
			pLB--; pLBdn--;
			ubPrev = *pUB;
			pUB--; pUBdn--;
		}


		pMapdn -= 2; pMap -= 2;
		pFeatdn -= 2; pFeat -= 2;
		pLBdn -= 2; pLB -= 2;
		pUBdn -= 2; pUB -= 2;
	}
}


cv::Mat fastMBS(const std::vector<cv::Mat> featureMaps)// 快速计算基于栅格扫描的显著性图。
{
	assert(featureMaps[0].type() == CV_8UC1);

	Size sz = featureMaps[0].size();
	Mat ret = Mat::zeros(sz, CV_32FC1);
	if (sz.width < 3 || sz.height < 3) 
		return ret;

	for (int i = 0; i < featureMaps.size(); i++)//对每个特征图（每个颜色通道）分别进行显著性检测。将所有特征图的显著性值累加到结果图 ret 中
	{
		Mat map = Mat::zeros(sz, CV_32FC1);
		Mat mapROI(map, Rect(1, 1, sz.width - 2, sz.height - 2));
		mapROI.setTo(Scalar(100000));
		Mat lb = featureMaps[i].clone();
		Mat ub = featureMaps[i].clone();

		rasterScan(featureMaps[i], map, lb, ub);
		invRasterScan(featureMaps[i], map, lb, ub);
		rasterScan(featureMaps[i], map, lb, ub);
		
		ret += map;
	}

	return ret;
	
	
}

float getThreshForGeo(const cv::Mat& src) {//计算用于地质测距显著性检测的阈值
    float ret = 0;
    Size sz = src.size();

    const uchar* pFeatup = src.data + 1;
    const uchar* pFeat = pFeatup + sz.width;
    const uchar* pFeatdn = pFeat + sz.width;

    int featPrev;

    for (int r = 1; r < sz.height - 1; r++) {
        featPrev = pFeat[-1];

        for (int c = 1; c < sz.width - 1; c++) {
            int temp = std::min(abs(pFeat[0] - featPrev), abs(pFeat[0] - pFeat[1]));
            temp = std::min(temp, std::min(abs(pFeat[0] - pFeatup[0]), abs(pFeat[0] - pFeatdn[0])));
            ret += temp;

            featPrev = pFeat[0];
            pFeat++;
            pFeatup++;
            pFeatdn++;
        }
        pFeat += 2;  // Skip the last pixel of the current row and the first pixel of the next row
        pFeatup += 2;
        pFeatdn += 2;
    }
    return ret / ((sz.width - 2) * (sz.height - 2));
	
}


void rasterScanGeo(const Mat& featMap, Mat& map, float thresh) {//进行正向栅格扫描，基于地质测距计算显著性
    Size sz = featMap.size();
    float *pMap = reinterpret_cast<float*>(map.data) + sz.width + 1;
    uchar *pFeat = featMap.data + sz.width + 1;

    for (int r = 1; r < sz.height - 1; r++) {
        float mapPrev = pMap[-1];
        float featPrev = pFeat[-1];

        for (int c = 1; c < sz.width - 1; c++) {
            float diffPrev = abs(featPrev - pFeat[c]) > thresh ? abs(featPrev - pFeat[c]) : 0.0f;
            float diffUp = abs(pFeat[c - sz.width] - pFeat[c]) > thresh ? abs(pFeat[c - sz.width] - pFeat[c]) : 0.0f;

            pMap[c] = std::min(pMap[c], mapPrev + diffPrev);
            pMap[c] = std::min(pMap[c], pMap[c - sz.width] + diffUp);

            mapPrev = pMap[c];
            featPrev = pFeat[c];
        }
        pMap += sz.width;
        pFeat += sz.width;
    }
}


void invRasterScanGeo(const Mat& featMap, Mat& map, float thresh) {//进行逆向栅格扫描，基于地质测距计算显著性
    Size sz = featMap.size();
    float *pMap = reinterpret_cast<float*>(map.data) + sz.width * (sz.height - 2) + sz.width - 2;
    uchar *pFeat = featMap.data + sz.width * (sz.height - 2) + sz.width - 2;

    for (int r = sz.height - 2; r > 0; r--) {
        float mapPrev = pMap[1];
        float featPrev = pFeat[1];

        for (int c = sz.width - 2; c > 0; c--) {
            float diffPrev = abs(featPrev - pFeat[c]) > thresh ? abs(featPrev - pFeat[c]) : 0.0f;
            float diffDown = abs(pFeat[c + sz.width] - pFeat[c]) > thresh ? abs(pFeat[c + sz.width] - pFeat[c]) : 0.0f;

            pMap[c] = std::min(pMap[c], mapPrev + diffPrev);
            pMap[c] = std::min(pMap[c], pMap[c + sz.width] + diffDown);

            mapPrev = pMap[c];
            featPrev = pFeat[c];
        }
        pMap -= sz.width;
        pFeat -= sz.width;
    }
}


cv::Mat fastGeodesic(const std::vector<cv::Mat> featureMaps)//快速计算基于地质测距的显著性图。
{
	assert(featureMaps[0].type() == CV_8UC1);

	Size sz = featureMaps[0].size();
	Mat ret = Mat::zeros(sz, CV_32FC1);
	if (sz.width < 3 || sz.height < 3)
		return ret;


	for (int i = 0; i < featureMaps.size(); i++)
	{
		// determines the threshold for clipping
		float thresh = getThreshForGeo(featureMaps[i]);
		//cout << thresh << endl;
		Mat map = Mat::zeros(sz, CV_32FC1);
		Mat mapROI(map, Rect(1, 1, sz.width - 2, sz.height - 2));
		mapROI.setTo(Scalar(1000000000));

		rasterScanGeo(featureMaps[i], map, thresh);
		invRasterScanGeo(featureMaps[i], map, thresh);
		rasterScanGeo(featureMaps[i], map, thresh);

		ret += map;
	}

	return ret;

}

int findFrameMargin(const Mat& img, bool reverse)
{
	Mat edgeMap, edgeMapDil, edgeMask;
	Sobel(img, edgeMap, CV_16SC1, 0, 1);
	edgeMap = abs(edgeMap);
	edgeMap.convertTo(edgeMap, CV_8UC1);
	edgeMask = edgeMap < (SOBEL_THRESH * 255.0);
	dilate(edgeMap, edgeMapDil, Mat(), Point(-1, -1), 2);
	edgeMap = edgeMap == edgeMapDil;
	edgeMap.setTo(Scalar(0.0), edgeMask);


	if (!reverse)
	{
		for (int i = edgeMap.rows - 1; i >= 0; i--)
			if (mean(edgeMap.row(i))[0] > 0.6*255.0)
				return i + 1;
	}
	else
	{
		for (int i = 0; i < edgeMap.rows; i++)
			if (mean(edgeMap.row(i))[0] > 0.6*255.0)
				return edgeMap.rows - i;
	}

	return 0;
}

bool removeFrame(const cv::Mat& inImg, cv::Mat& outImg, cv::Rect &roi)
{
	if (inImg.rows < 2 * (FRAME_MAX + 3) || inImg.cols < 2 * (FRAME_MAX + 3))
	{
		roi = Rect(0, 0, inImg.cols, inImg.rows);
		outImg = inImg;
		return false;
	}

	Mat imgGray;
	cvtColor(inImg, imgGray, cv::COLOR_RGB2GRAY);

	int up, dn, lf, rt;
	
	up = findFrameMargin(imgGray.rowRange(0, FRAME_MAX), false);
	dn = findFrameMargin(imgGray.rowRange(imgGray.rows - FRAME_MAX, imgGray.rows), true);
	lf = findFrameMargin(imgGray.colRange(0, FRAME_MAX).t(), false);
	rt = findFrameMargin(imgGray.colRange(imgGray.cols - FRAME_MAX, imgGray.cols).t(), true);

	int margin = MAX(up, MAX(dn, MAX(lf, rt)));
	if ( margin == 0 )
	{
		roi = Rect(0, 0, imgGray.cols, imgGray.rows);
		outImg = inImg;
		return false;
	}

	int count = 0;
	count = up == 0 ? count : count + 1;
	count = dn == 0 ? count : count + 1;
	count = lf == 0 ? count : count + 1;
	count = rt == 0 ? count : count + 1;

	// cut four border region if at least 2 border frames are detected
	if (count > 1)
	{
		margin += 2;
		roi = Rect(margin, margin, inImg.cols - 2*margin, inImg.rows - 2*margin);
		outImg = Mat(inImg, roi);

		return true;
	}

	// otherwise, cut only one border
	up = up == 0 ? up : up + 2;
	dn = dn == 0 ? dn : dn + 2;
	lf = lf == 0 ? lf : lf + 2;
	rt = rt == 0 ? rt : rt + 2;

	
	roi = Rect(lf, up, inImg.cols - lf - rt, inImg.rows - up - dn);
	outImg = Mat(inImg, roi);

	return true;
	
}


Mat doWork(
    const Mat& src,
    bool use_lab,
    bool remove_border,
    bool use_geodesic
)
{
    Mat src_small;
    float w = (float)src.cols, h = (float)src.rows;
    float maxD = max(w, h);
    resize(src, src_small, Size((int)(MAX_IMG_DIM * w / maxD), (int)(MAX_IMG_DIM * h / maxD)), 0.0, 0.0, INTER_AREA); // standard: width: 300 pixel INTER_AREA会减小噪声

    Mat srcRoi;
    Rect roi;
    // detect and remove the artificial frame of the image
    if (remove_border)
        removeFrame(src_small, srcRoi, roi);
    else
    {
        srcRoi = src_small;
        roi = Rect(0, 0, src_small.cols, src_small.rows);
    }

    if (use_lab) {cvtColor(srcRoi, srcRoi, cv::COLOR_RGB2Lab);}
    /* Computing saliency */
    MBS mbs(srcRoi);
    mbs.computeSaliency(use_geodesic);
    Mat result = mbs.getSaliencyMap();
	// showImageWithConfig("no post-processing image", result, 0, 0, 200, 200);
	
    /* Post-processing */
	Scalar s = mean(result);
	int radius = floor(50 * sqrt(s.val[0]/255.0));
	radius = (radius > 3) ? radius : 3;//warehaouse
	// radius = (radius > 5) ? radius : 5;

	// int radius=3;
	Mat morphmat = morpySmooth(result, radius);
	Mat smallResult =enhanceConstrast(morphmat,3);
	// Mat smallResult =enhanceContrastCLAHE(morphmat);

	Mat finalResult;
	resize(smallResult, finalResult, src.size());
	// showImageWithConfig("smooth-processing image", morphmat, 600, 0, 200, 200);
	// showImageWithConfig("Contrast enhanced image", smallResult, 600, 0, 200, 200);
    return finalResult;

	
}

Mat morpySmooth(Mat I, int radius) 
{
    Mat se = getStructuringElement(MORPH_ELLIPSE, Size(radius, radius));
    Mat openmat, recon1, dilatemat, recon2, res;

    morphologyEx(I, openmat, MORPH_OPEN, se, Point(-1, -1),1);//开运算去除噪声点，闭运算填补空洞，平滑物体边界 warehouse

    // Perform morphological reconstruction
    recon1 = openmat.clone();
    Mat se1 = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    Mat tmp1, tmp2;
    do
    {
        recon1.copyTo(tmp1);
        dilate(recon1, tmp2, se , Point(-1, -1),1);//对图像进行膨胀操作 填充小孔和连接断裂的物体
        cv::min(I, tmp2, recon1);
    } while (cv::norm(tmp1, recon1, NORM_INF) != 0);

    // // Dilate the reconstructed image
    // dilate(recon1, dilatemat, se1,Point(-1, -1),1);

    // // Prepare for second reconstruction
    // recon1 = 255 - recon1;
    // dilatemat = 255 - dilatemat;

    // // Perform second morphological reconstruction
    // recon2 = dilatemat.clone();
    // do
    // {
    //     recon2.copyTo(tmp1);
    //     dilate(recon2, tmp2, se);
    //     cv::min(recon1, tmp2, recon2);
    // } while (cv::norm(tmp1, recon2, NORM_INF) != 0);

    // // Invert the final reconstruction to get the result
    // res = 255 - recon2;

    // return res;
    return recon1;


}

cv::Mat enhanceContrastCLAHE(const cv::Mat& input)//CLAHE 是一种自适应直方图均衡化方法，通过对局部图像块进行均衡化，防止过度增强，同时增强对比度。避免噪声放大
{
    cv::Mat result;
    
    if (input.channels() == 1) {
        // 如果是灰度图，直接应用 CLAHE
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->setClipLimit(2.0);//设置对比度限制。较低的可以避免噪声的放大，对比度增强效果较弱。较高的 clipLimit可以显著增强对比度，突出图像细节和显著区域。可能会放大噪声。
        clahe->apply(input, result);
		cv::imshow("CLACHE enhanced image",result);
		cv::threshold(result, result, 10, 255, cv::THRESH_BINARY );
    } 
	else {
        throw std::invalid_argument("Unsupported number of channels: " + std::to_string(input.channels()));
    }

    return result;
}

Mat enhanceConstrast(Mat I, int b) // 默认参数在定义时不提供
{
    I.convertTo(I, CV_32FC1);
    int total = I.rows * I.cols, num1(0), num2(0);
    double max, min, t, sum1(0), sum2(0), v1, v2,midVal;
    Point p1, p2;
    minMaxLoc(I, &min, &max, &p1, &p2);
    t = max * 0.5;

    for (int i = 0; i < I.rows; i++)
    {
        float* indata = I.ptr<float>(i);
        for (int j = 0; j < I.cols; j++)
        {
            float temp = indata[j];
            if (temp >= t)
            {
                sum1 += temp;
                num1++;
            }
            else
            {
                sum2 += temp;
                num2++;
            }
        }
    }
    v1 = sum1 / num1; // Average of higher intensity values
    v2 = sum2 / num2; // Average of lower intensity values
    midVal = 0.5 * (v1 + v2);

	// Use a lookup table for caching
    unordered_map<int, float> cache;
    double factor = -b; // Precompute factor

    // Create a lookup table for exp function
    for (int i = 0; i <= 255; i++) 
    {
        double x = (i - midVal) * factor;
        cache[i] = 1.0 / (exp(x) + 1) * 255.0;
    }

    // Second pass: Apply contrast enhancement using cache
    for (int i = 0; i < I.rows; i++)
    {
        float* indata = I.ptr<float>(i);
        for (int j = 0; j < I.cols; j++)
        {
            int temp = static_cast<int>(indata[j]);
            if (cache.find(temp) != cache.end())
            {
                indata[j] = cache[temp];
            }
            else
            {
                // Fallback if out of bounds (though should not happen)
                indata[j] = 1.0 / (exp((temp - midVal) * factor) + 1) * 255.0;
            }
        }
    }
    return I;
}

void showImageWithConfig(const std::string& windowName, const cv::Mat& image, int x, int y, int width, int height)
{
    cv::imshow(windowName, image);
    cv::moveWindow(windowName, x, y);
    cv::resizeWindow(windowName, width, height);
}
class BestOf2NearestMatcher {
    constructor() {
        this.matchConf = 0;
        this.numMatchesThresh1 = 0;
        this.numMatchesThresh2 = 0;
    }

    genMatrix(height, width, val) {
        if (height === undefined || height < 1 || width < 1) {
            return [];
        } else if (width === undefined) {
            let arr1D = [];
            for (let i = 0; i < height; i++) {
                arr1D.push(val);
            }
            return arr1D;
        } else {
            let arr1D = [];
            let arr2D = [];
            for (let i = 0; i < width; i++) {
                arr1D.push(val);
            }
            for (let i = 0; i < height; i++) {
                arr2D.push(arr1D.slice());
            }
            return arr2D;
        }
    }

    mat2Array(mat) {
        const height = src.length;
        const width = src[0].length;
        let arr = this.genArray(height * width, 0);
        for (let i = 0; i < height; i++) {
            for (let j = 0; j < width; j++) {
                arr[i * width + j] = mat[i][j];
            }
        }
        return arr;
    }

    checkVector(mat, queryDims) {
        const elements = mat.length;
        const dims = mat[0].length;
        return queryDims === dims ? elements : -1;
    }

    convertPointsFromHomogeneous(src, dst) {
        let i = 0;
        let cn = 0;
        let npoints = this.checkVector(src, 3);
        cn = 3;
        if (npoints < 0) {
            npoints = this.checkVector(src, 4);
            cn = 4; 
        }

        dst = this.genMatrix(npoints, 1, 0);
        if (cn === 3) {
            for (let i = 0; i < npoints; i++) {
                let scale = src[i][2] !== 0 ? 1.0 / src[i][2] : 1.0;
                dst[i][0] = src[i][0] * scale;
                dst[i][1] = src[i][1] * scale;
            }
        } else {
            let scale = src[i][3] !== 0 ? 1.0 / src[i][3] : 1.0;
            dst[i][0] = src[i][0] * scale;
            dst[i][1] = src[i][1] * scale;
            dst[i][2] = src[i][2] * scale;
        }
    }

    // reshape(mat, newChannel, newHeight) {
    reshape(mat, newHeight, newWidth) {
        const arr = this.mat2Array(mat);
        let res = this.genMatrix(newHeight, newWidth);
        let k = 0;
        for (let i = 0; i < newHeight; i++) {
            for (let j = 0; j < newWidth; j++) {
                res[i][j] = arr[k];
                k += 1;
            }
        }
        return res;
    }

    compressElems(src, mask, mstep, count) {
        let i = 0;
        let j = 0;
        for (i = 0; i < count; i++) {
            if (mask[i * mstep]) {
                if(i > j) {
                    src[j] = src[i];
                }
                j += 1;
            }
        }
        return j;
    }

    rowRange(mat, startRow, endRow) {
        const width = mat[0].length;
        const newHeight = endRow - startRow;
        let res = this.genMatrix(newHeight, width, 0);
        for (let i = 0; i < newHeight; i++) {
            for (let j = 0; j < width; j++) {
                res[i][j] = mat[i + startRow][j];
            }
        }
        return res;
    }

    findHomography(points1, points2, method, ransacReprojThreshold, mask, maxIters, confidence) {
        const defaultRANSACReprojThreshold = 3;
        let result = false;
        let npoints = -1;
        let tempMask = [];
        let src = [];
        let dst = [];
        let H = [];

        for (let i = 1; i <= 2; i++) {
            let p = i === 1 ? points1 : points2;
            let m = i === 1 ? src : dst;
            // npoints = this.checkVector(p, 2, -1, false);
            npoints = this.checkVector(p, 2);
            if (npoints < 0) {
                // npoints = this.checkVector(p, 3, -1, false);
                npoints = this.checkVector(p, 3);
                if (npoints <= 0) {
                    return [];
                }
                this.convertPointsFromHomogeneous(p, p);
            }
            // m = this.reshape(p, 2, npoints);
            m = this.reshape(p, npoints, 2);
        }
        if (ransacReprojThreshold <= 0) {
            ransacReprojThreshold = defaultRANSACReprojThreshold;
        }
        let cb = new HomographyEstimatorCallback();
        if (method === 0 || npoints === 4) {
            tempMask = this.genMatrix(npoints, 1, 1);
            result = cb.runKernel(src, dst, H) > 0;
        } else if (method === RANSAC) {
            // result = this.createRANSACPointSetRegistrator(cb, 4, ransacReprojThreshold, confidence, maxIters).run(src, dst, H, tempMask);
            let ransac = new RANSAC();
            result = ransac.run(src, dst, H, tempMask);
        } else if (method === LMEDS) {
            //TODO
        } else if (method === RHO) {
            //TODO
        } else {
            return;
        }

        if (result && npoints > 4 && method != RHO) {
            this.compressElems(src, tempMask, 1, npoints);
            npoints = this.compressElems(dst, tempMask, 1, npoints);
            if (npoints > 0) {
                let src1 = rowRange(src, 0, npoints);
                let dst1 = rowRange(dst, 0, npoints);
                src = src1;
                dst = dst1;
                if (method == RANSAC || method ==LMEDS) {
                    cb.runKernel(src, dst, H);
                }
                let H8 = this.genMatrix(8, 1, 0);
                // TODO LMSolver
                let lmSolver = new LMSolver();
                let homographyRefine = new HomographyRefine(src, dst);
                lmSolver.create(homographyRefine, 10);
                lmSolver.run(H8);

                H = H8;
            }
        }
        // TODO _mask.needed()
        return H;
    }

    determinant() {
    }

    match(features1, features2, matchesInfo) {
        if (matchesInfo.matches.length < this.numMatchesThresh1) {
            return;
        }

        let srcPoints = this.genMatrix(matchesInfo.matches.length, 2, 0);
        let dstPoints = this.genMatrix(matchesInfo.matches.length, 2, 0);
        for (let i = 0; i < matchesInfo.matches.length; i++) {
            const m = matchesInfo.matches[i];
            let px = features1.keypoints[m.queryIdx].x;
            let py = features1.keypoints[m.queryIdx].y;
            px -= features1.width * 0.5;
            py -= features1.height * 0.5;
            srcPoints[i][0] = px;
            srcPoints[i][1] = py;

            px = features2.keypoints[m.queryIdx].x;
            py = features2.keypoints[m.queryIdx].y;
            px -= features2.width * 0.5;
            py -= features2.height * 0.5;
            dstPoints[i][0] = px;
            dstPoints[i][1] = py;
        }
        matchesInfo.H = this.findHomography(srcPoints, dstPoints, matchesInfo.inliersMask, 'RANSAC');
        if (!matchesInfo.H || Math.abs(this.determinant(matchesInfo.H)) < 1e-6) {
            return;
        }
        matchesInfo.numInliers = 0;
        for (let i = 0; i < matchesInfo.inliersMask.length; i++) {
            if (matchesInfo,inliersMask[i]) {
                matchesInfo.numInliers += 1;
            }
        }
        matchesInfo.confidece = matchesInfo.numInliers / (8 + 0.3 * matchesInfo.matches.length);
        matchesInfo.confidece = matchesInfo.confidence > 3 ? 0.0 : matchesInfo.confidence;
        if (matchesInfo.numInliers < numMatchesThresh2) {
            return;
        }

        // for inliers
        // TODO create
        // srcPoints.create(1, matchesInfo.numInliers);
        // dstPoints.create(1, matchesInfo.numInliers);
        srcPoints = this.genMatrix(matchesInfo.numInliers, 2, 0);
        dstPoints = this.genMatrix(matchesInfo.numInliers, 2, 0);
        let inlierIdx = 0;
        for (let i = 0; i < matchesInfo.matches.length; i++) {
            if (!matchesInfo.inliersMask[i]) {
                continue;
            }
            const m = machesInfo.matches[i];
            let px = features1.keypoints[m.queryIdx].x;
            let py = features1.keypoints[m.queryIdx].y;
            px -= features1.width * 0.5;
            py -= features1.height * 0.5;
            srcPoints[i][0] = px;
            srcPoints[i][1] = py;

            px = features2.keypoints[m.queryIdx].x;
            py = features2.keypoints[m.queryIdx].y;
            px -= features2.width * 0.5;
            py -= features2.height * 0.5;
            dstPoints[i][0] = px;
            dstPoints[i][1] = py;
            inlierIdx += 1;
        }
        matchesInfo.H = this.findHomography(srcPoints, destPoints, 'RANSAC');
    }

    operator(features, pairMatches, mask, rangeWidth) {
        const numImages = features.length;
        let mask_ = this.genMatrix(numImages, numImages, 1);
        let nearPairs = new Pair();
        for (let i = 0; i < numImages - 1; i++) {
            for (let j = i + 1; j < Math.min(numImages, i + rangeWidth); j++) {
                if (features[i].keypoints.length > 0 && features[j].keypoints.length > 0 && mask_[i][j]) {
                    nearPairs.push(this.makePair(i, j));
                }
            }
        }
        pairMatches.resize(numImages * numImages);
        let body = new MatchPairsBody(this, features, pairMatches, nearPairs);
        // TODO
        body(0, nearPairs.length);
    }

    collectGarbage() {
    }
}

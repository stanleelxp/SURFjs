const { inverse } = require('ml-matrix');

class BundleAdjusterRay () {
    constructor() {
        this.numImages = 0;
        this.totalNumMatches = 0;
        this.err1 = 0;
        this.err2 = 0;
        this.cameraParams = [];
        this.edges = {};
        this.features = {};
        this.pairMatches = [];
    }

    genArray(height, val) {
    }

    genMatrix(height, width, val) {
    }

    genEye(n) {
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

    array2Mat(arr) {
        const len = arr.length;
        const width = Math.floor(Math.sqrt(len));
        const height = Math.ceil(1.0 * len / width);
        let mat = this.genArray(height * width, 0);
        for (let i = 0; i < height; i++) {
            for (let j = 0; j < width; j++) {
                mat[i][j] = arr[i * width + j];
            }
        }
        return mat;
    }

    scalarMulArray(scalar, arr) {
        const height = arr.length;
        let res = this.genArray(height, 0);
        for (let i = 0; i < height; i++) {
            res[i] = scalar * arr[i];
        }
        return res;
    }

    scalarMulMatrix(scalar, mat) {
        const height = mat.length;
        const width = mat[0].length;
        let res = this.genMatrix(height, width, 0);
        for (let i = 0; i < height; i++) {
            for (let j = 0; j < width; j++) {
                res[i][j] = scalar * mat[i][j];
            }
        }
        return res;
    }

    norm(arr) {
        // L2 norm
        const len = arr.length;
        let sum = 0;
        for (let i = 0; i < len; i++) {
           sum += arr[i] * arr[i]; 
        }
        return Math.sqrt(sum);
    }

    inRange(src, rMin, rMax) {
        const height = src.length;
        const width = src[0].length;
        if (width) {
            for (let i = 0; i < height; i++) {
                for (let j = 0; j < width; j++) {
                    if (src[i][j] < rMin || src[i][j] > rMax) {
                        return false;
                    } else {
                        return true;
                    }
                }
            }
        } else {
            for (let i = 0; i < height; i++) {
                if (src[i] < rMin || src[i] > rMax) {
                    return false;
                } else {
                    return true;
                }
            }
        }
    }

    // opencv modules/calib3d/src/calibration.cpp
    Rodrigues(src, dst) {
        const DBL_EPSILON = 1e-3;
        const height = src.length;
        const width = src[0].length;
        let srcArray = this.mat2Array(src);
        if (width === 1 || height === 1) {
            let r = [srcArray[0], srcArray[1], srcArray[2]];
            let theta = this.norm(r);
            if (theta < DBL_EPSILON) {
                dst = this.genEye(3);
            } else {
                let c = Math.cos(theta);
                let s = Math.sin(theta);
                let c1 = 1.0 - c;
                let itheta = theta ?  1.0 / theta : 0.0;
                r[0] *= itheta;
                r[1] *= itheta;
                r[2] *= itheta;
                let rrt = [[r[0] * r[0], r[0] * r[1], r[0] * r[2]],
                    [r[0] * r[1], r[1] * r[1], r[1] * r[2]],
                    [r[2] * r[1], r[2] * r[1], r[2] * r[2]]];

                let rx = [[0, -r[2], r[1]],
                    [r[2], 0, -r[0]],
                    [-r[1], r[0], 0]];

                let I3 = this.genEye(3);
                dst = this.scalarMulMatrix(c, I3) + this.scalarMulMatrix(c1, rrt) + this.scalarMulMatrix(s, rx); 
            }
        } else if (width === 3 && height === 3) {
            let R = src;
            if (!this.inRange(R, -100, 100)) {
                dst = this.genMatrix(3, 1, 0);
                return 0;
            }

            // TODO SVD
            // SVD.compute(R, W, U, Vt);
            // R = U * Vt;
            //
            let r = [R[2][1] - R[1][2], R[0][2] - R[2][0], R[1][0], R[0][1]];
            let s = Math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2] ) * 0.25); 
            let c = (R[0][0] + R[1, 1] + R[2, 2] - 1) * 0.5;
            if (c > 1.0) {
                c = 1.0;
            }
            if (c < -1.0) {
                c =-1.0;
            }
            theta = Math.acos(c);
            if (s < 1e-5) {
                let t = 0;
                if (c > 0) {
                    r = [0, 0, 0];
                } else {
                    t = (R[0][0] + 1) * 0.5;
                    r[0] = Math.sqrt(math.max(t, 0)); 
                    t = (R[1][1] + 1) * 0.5;
                    r[1] = Math.sqrt(math.max(t, 0)) * (R[0][1] < 0 ? -1 : 1); 
                    t = (R[2][2] + 1) * 0.5;
                    r[2] = Math.sqrt(math.max(t, 0)) * (R[0][2] < 0 ? -1 : 1); 
                    if (Math.abs(r[0]) < Math.abs(r[1]) && Math.abs(r[0]) < Math.abs(r[2]) && (R[1][2] > 0) != (r[1] * r[2] > 0)) {
                        r[2] *= -1;
                    }
                    theta /= this.norm(r);
                    r[0] *= theta;
                    r[1] *= theta;
                    r[2] *= theta;
                }
            } else {
                let vth = 0.5 / s;

                // if (jac) {}
                //
                vth *= theta;
                r[0] *= vth;
                r[1] *= vth;
                r[2] *= vth;
            }
            dst = r;
        }
        return 1;
    }

    setUpInitialCameraParams(cameras) {
        this.cameraParams = this.genMatrix(this.numImages * 4, 1, 0);
        for (let i = 0; i < this.numImages; i++ ) { 
            let R = [];
            // SVD
            // svd(cameras[i].R, SVD::FULL_UV);
            // R = svd.u * svd.vt;
            // if (det(R) < 0) {
            //     R *= -1;
            // }
            let rvec = [];
            this.Rodrigues(R, rvec);
            this.cameraParams[i * 4 + 1][0] = rvec[0][0];
            this.cameraParams[i * 4 + 2][0] = rvec[1][0];
            this.cameraParams[i * 4 + 3][0] = rvec[2][0];
        }
    }

    obtainRefinedCameraParams(cameras) {
        for (let i = 0; i < this.numImages; i++) {
            cameras[i].focal = this.cameraParams[i * 4][0];
            let rvec = this.genMatrix(3, 1, 0);
            rvec[0][0] = this.cameraParams[i * 4 + 1][0];
            rvec[1][0] = this.cameraParams[i * 4 + 2][0];
            rvec[2][0] = this.cameraParams[i * 4 + 3][0];
            this.Rodrigues(rvec, cameras[i].R);
        }
    }
    
    calcError(err) {
        err = this.genMatrix(this.totalNumMatches * 3, 1);
        let matchIdx = 0;
        for (let edgeIdx = 0; edgeIdx < this.edges.length; edgeIdx++) {
            let i = this.edges[edgeIdx][0];
            let j = this.edges[edgeIdx][0];
            let f1 = this.cameraParams[i * 4][0];
            let f2 = this.cameraParams[j * 4][0];
            
            let R1 = this.genArray(9, 0);
            let R1_  = this.genMatrix(3, 3, 0);
            let rvec = this.genMatrix(3, 1, 0);
            rvec[0][0] = this.cameraParams[i * 4 + 1][0];
            rvec[1][0] = this.cameraParams[i * 4 + 2][0];
            rvec[2][0] = this.cameraParams[i * 4 + 3][0];
            this.Rodrigues(rvec, R1_);
            
            let R2 = this.genArray(9, 0);
            let R2_ = this.genMatrix(3, 3 ,0);
            rvec[0][0] = this.cameraParams[j * 4 + 1][0];
            rvec[1][0] = this.cameraParams[j * 4 + 2][0];
            rvec[2][0] = this.cameraParams[j * 4 + 3][0];
            this.Rodrigues(rvec, R2_);

            const features1 = this.features[i];
            const features2 = this.features[j];
            const matchesInfo = this.pairMatches[i * this.numImages + j];
            
            let K1 = this.genEye(3);
            K1[0][0] = f1;
            K1[0][2] = features1.imgSize.width * 0.5;
            K1[1][1] = features1.imgSize.height * 0.5;
            let K2 = this.genEye(3);
            K2[0][0] = f2;
            K2[0][2] = features2.imgSize.width * 0.5;
            K2[1][1] = features2.imgSize.height * 0.5;
            
            let R1m = new Matrix(R1_);
            let R2m = new Matrix(R2_);
            let H1 = R1m.mmul(inverse(K1)); 
            let H2 = R2m.mmul(inverse(K2)); 

            for (let k = 0; k < matchesInfo.matches.length; k++) {
                if (!matchesInfo.inliersMask[k]) { 
                    continue;
                }
                const m = matchesInfo.matches[k];
                let p1 = features1.keypoints[m.queryIdx].pt;
                let x1 = H1[0][0] * p1.x + H1[0][1] * p1.y + H1[0][2];
                let x1 = H1[1][0] * p1.x + H1[1][1] * p1.y + H1[1][2];
                let z1 = H1[2][0] * p1.x + H1[2][1] * p1.y + H1[2][2];
                let len = Math.sqrt(x1 * x1 + y1 * y1 + z1 * z1);
                x1 /= len;
                y1 /= len;
                z1 /= len;

                let p2 = features2.keypoints[m.queryIdx].pt;
                let x2 = H2[0][0] * p2.x + H2[0][1] * p2.y + H2[0][2];
                let x2 = H2[1][0] * p2.x + H2[1][1] * p2.y + H2[1][2];
                let z2 = H2[2][0] * p2.x + H2[2][1] * p2.y + H2[2][2];
                let len = Math.sqrt(x2 * x2 + y2 * y2 + z2 * z2);
                x2 /= len;
                y2 /= len;
                z2 /= len;

                let mult = Math.sqrt(f1 * f2);
                err[3 * matchIdx + 0][0] = mult * (x1 - x2);
                err[3 * matchIdx + 1][0] = mult * (y1 - y2);
                err[3 * matchIdx + 2][0] = mult * (z1 - z2);

                matchIdx += 1;
            }
        }
    }

    calcDeriv(err1, err2, h, col, res) {
        for (let i = 0; i < err1.length; i++) {
            // res[i][0] = (err2[i][0] - err1[i][0]) / h; 
            res[i][col] = (err2[i][col] - err1[i][col]) / h; 
        }
    }

    calcJacobian(jac) {
        jac = this.genMatrix(this.totalNumMatches * 3, this.numImages * 4);
        let val = 0;
        const step = 1e-3;
        for (let i = 0; i < this.numImages; i++) {
            for (let j = 0; j < 4; j++) {
                val = this.cameraParams[i * 4 + j][0];
                this.cameraParams[i * 4 + j][0] = val - step;
                this.calcError(this.err1);
                this.cameraParams[i * 4 + j][0] = val + step;
                this.calcError(this.err2);
                // TODO col
                // this.calcDeriv(this.err1, this.err2, 2 * step, jac.col(i * 4 + j));
                this.calcDeriv(this.err1, this.err2, 2 * step, i * 4 + j, jac);
                this.cameraParams[i * 4 + j][0] = val;
            }
        }
    }
}


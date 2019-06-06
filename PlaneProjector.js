const { Matrix, inverse } = require('ml-matrix'); // ~6.0.0


class PlaneProjector() {
    constructor() {
        this.rinv = [];
        this.rkinv = [];
        this.krinv = [];
        this.k = this.genArray(9, 0);
        this.t = this.genArray(3, 0);
        this.scale = 1.0;

    }

    isNumber(obj) {
        return obj === +obj;
    }

    genArray(height, val) {
        if (height === undefined || height < 1 || !isNumber(val)) {
            return [];
        } else {
            let arr1D = [];
            for (let i = 0; i < height; i++) {
                arr1D.push(val);
            }
            return arr1D;
        }
    }

    genMatrix(height, width, val) {
        if (height === undefined || height < 1 || width < 1 || !isNumber(val)) {
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

    mapForward(x, y, u, v) {
        rkinv = this.rkinv;
        scale = this.scale;
        t = this.t;
        let x_ = rkinv[0] * x + rkinv[1] * y + rkinv[2]; 
        let y_ = rkinv[3] * x + rkinv[4] * y + rkinv[5]; 
        let x_ = rkinv[6] * x + rkinv[7] * y + rkinv[8]; 
        x_ = t[0] + x_ / z_ * (1 - t[2]);
        y_ = t[1] + y_ / z_ * (1 - t[2]);

        u = scale * x_;
        v = scale * y_;


    }

    mapBackward(u, v, x, y) {
        krinv = this.krinv;
        scale = this.scale;
        t = this.t;
        u = u / scale - t[0];
        v = v / scale - t[1];
        let  z = 0;
        x = krinv[0] * u + krinv[1] * v + krinv[2] * (1 - t[2]); 
        y = krinv[3] * u + krinv[4] * v + krinv[5] * (1 - t[2]); 
        z = krinv[6] * u + krinv[7] * v + krinv[8] * (1 - t[2]); 

        x /= z;
        y /= z;
    }

    setCameraParams(K, R, T) {
        this.k = [K[0][0], K[0][1], K[0][2],
            K[1][0], K[1][1], K[1][2],
            K[2][0], K[2][1], K[2][2]];

        // let Rinv = matrix.transpose(R); 
        const R_ = new Matrix(R);
        const K_ = new Matrix(K);
        const Rinv = R_.transpose(); 
        const Kinv = inverse(K_);
        const RKinv = R_.mmul(Kinv); 
        const KRinv = K_.mmul(Rinv);

        this.rinv = [Rinv[0][0], Rinv[0][1], Rinv[0][2],
            Rinv[1][0], Rinv[1][1], Rinv[1][2],
            Rinv[2][0], Rinv[2][1], Rinv[2][2]];

        this.krinv = [KRinv[0][0], KRinv[0][1], KRinv[0][2],
            KRinv[1][0], KRinv[1][1], KRinv[1][2],
            KRinv[2][0], KRinv[2][1], KRinv[2][2]];
        
        this.t = [T[0][0], T[0][1], T[0][2]];
    }
}


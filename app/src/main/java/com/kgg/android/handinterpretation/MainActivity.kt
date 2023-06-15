package com.kgg.android.handinterpretation

import android.Manifest
import android.app.Activity
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.core.app.ActivityCompat
import androidx.core.content.ContentProviderCompat.requireContext
import com.google.mediapipe.solutioncore.CameraInput
import com.google.mediapipe.solutioncore.SolutionGlSurfaceView
import com.google.mediapipe.solutions.hands.Hands
import com.google.mediapipe.solutions.hands.HandsOptions
import com.google.mediapipe.solutions.hands.HandsResult
import com.gun0912.tedpermission.PermissionListener
import com.gun0912.tedpermission.normal.TedPermission
import com.kgg.android.handinterpretation.databinding.ActivityMainBinding
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.acos
import kotlin.math.round
import kotlin.math.sqrt

class MainActivity : AppCompatActivity() {

    private lateinit var hands : Hands
    private lateinit var cameraInput: CameraInput
    private lateinit var glSurfaceView: SolutionGlSurfaceView<HandsResult>

    // 필요 권한 확인
    private val REQUIRED_PERMISSIONS = mutableListOf(Manifest.permission.INTERNET,
        Manifest.permission.RECORD_AUDIO, Manifest.permission.CAMERA,
        Manifest.permission.ACCESS_NETWORK_STATE).toTypedArray()

    // 전역 변수로 바인딩 객체 선언
    private var mBinding: ActivityMainBinding? = null

    // 뷰 초기화
    fun initView() {
        mBinding = ActivityMainBinding.inflate(layoutInflater)
    }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        mBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(mBinding!!.root)

        // 권한 확인
        val permissionListener = object : PermissionListener {
            override fun onPermissionGranted() {
                setupStreamingModePipeline()

                glSurfaceView.post { startCamera() }
                glSurfaceView.visibility = View.VISIBLE

                initView()
            }

            override fun onPermissionDenied(deniedPermissions: MutableList<String>?) {
                Toast.makeText(this@MainActivity, "해당 관한이 없습니다. : " + deniedPermissions, Toast.LENGTH_SHORT).show()
            }

        }
        setupStreamingModePipeline()
        glSurfaceView.post { startCamera() }
        glSurfaceView.visibility = View.VISIBLE


        TedPermission.create()
            .setPermissionListener(permissionListener)
            .setPermissions(*REQUIRED_PERMISSIONS)
            .check()


    }

    /**
     * Describe : MediaPipe Hands 라이브러리를 사용하여 손 추적 시스템을 설정하고 Android 애플리케이션에서 GLSurfaceView에 결과를 표시
     */
    private fun setupStreamingModePipeline() {

        // Mediapipe의 손 동작 감지 객체
        hands = Hands(
            this@MainActivity,
            HandsOptions.builder()
                .setStaticImageMode(false)
                .setMaxNumHands(1)
                .setRunOnGpu(true)
                .build()
        )
        hands.setErrorListener { message, e -> Log.e("TAG", "MediaPipe Hands error: $message") }

        //  새로운 카메라 프레임을 수신하기 위한 리스너를 설정
        cameraInput = CameraInput(this@MainActivity)
        cameraInput.setNewFrameListener { hands.send(it) }

        glSurfaceView = SolutionGlSurfaceView(this@MainActivity, hands.glContext,
            hands.glMajorVersion)
        glSurfaceView.setSolutionResultRenderer(HandsResultGlRenderer())
        glSurfaceView.setRenderInputImage(true)

        // 추적 준비가 되었을 때 동작
        hands.setResultListener {
            translate(it)
            glSurfaceView.setRenderData(it)
            glSurfaceView.requestRender()
        }

        glSurfaceView.post(this::startCamera)

        // activity_main.xml에 선언한 FrameLayout에 화면을 띄우는 코드
        mBinding!!.previewDisplayLayout.apply {
            removeAllViewsInLayout()
            addView(glSurfaceView)
            glSurfaceView.visibility = View.VISIBLE
            requestLayout()
        }
    }

    private fun startCamera() {
        cameraInput.start(
            this@MainActivity,
            hands.glContext,
            CameraInput.CameraFacing.FRONT,
            glSurfaceView.width,
            glSurfaceView.height
        )
    }


    // tfilte

    private fun getTfliteInterpreter(path: String): Interpreter? {
        try {
            return Interpreter(loadModelFile(this@MainActivity, path)!!)
        }
        catch (e: Exception) {
            e.printStackTrace()
        }
        return null
    }

    private fun loadModelFile(activity: Activity, path: String): MappedByteBuffer? {
        val fileDescriptor = activity.assets.openFd(path)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }


    private fun translate(result : HandsResult){
        if (result.multiHandLandmarks().isEmpty()) {
            return
        }
        val landmarkList = result.multiHandLandmarks()[0].landmarkList
        val joint = Array(21){FloatArray(3)}
        for(i in 0..19) {
            joint[i][0] = landmarkList[i].x
            joint[i][1] = landmarkList[i].y
            joint[i][2] = landmarkList[i].z
        }

        val v1 = joint.slice(0..19).toMutableList()
        for(i in 4..16 step(4)) {
            v1[i] = v1[0]
        }
        var v2 = joint.slice(1..20)
        val v = Array(20) { FloatArray(3) }

        for(i in 0..19) {
            for(j in 0..2) {
                v[i][j] = v2[i][j] - v1[i][j]
            }
        }

        for(i in 0..19) {
            val norm = sqrt(v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2])
            for(j in 0..2) {
                v[i][j] /= norm
            }
        }

        val tmpv1 = mutableListOf<FloatArray>()
        for(i in 0..18) {
            if(i != 3 && i != 7 && i != 11 && i != 15) {
                tmpv1.add(v[i])
            }
        }
        val tmpv2 = mutableListOf<FloatArray>()
        for(i in 1..19) {
            if(i != 4 && i != 8 && i != 12 && i != 16) {
                tmpv2.add(v[i])
            }
        }

        val einsum = FloatArray(15)
        for( i in 0..14) {
            einsum[i] = tmpv1[i][0] * tmpv2[i][0] + tmpv1[i][1] * tmpv2[i][1] +
                    tmpv1[i][2] * tmpv2[i][2]
        }
        val angle = FloatArray(15)
        val data = FloatArray(15)
        for(i in 0..14) {
            angle[i] = Math.toDegrees(acos(einsum[i]).toDouble()).toFloat()
            data[i] = round(angle[i] * 100000) / 100000
        }

        val interpreter = getTfliteInterpreter("model.tflite")
        val byteBuffer = ByteBuffer.allocateDirect(15*4).order(ByteOrder.nativeOrder())

        for(d in data) {
            byteBuffer.putFloat(d)
        }

        val modelOutput = ByteBuffer.allocateDirect(26*4).order(ByteOrder.nativeOrder())
        modelOutput.rewind()

        interpreter!!.run(byteBuffer,modelOutput)

        val outputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1,26), DataType.FLOAT32)
        outputFeature0.loadBuffer(modelOutput)

        // ByteBuffer to FloatBuffer
        val outputsFloatBuffer = modelOutput.asFloatBuffer()
        val outputs = mutableListOf<Float>()
        for(i in 1..26) {
            outputs.add(outputsFloatBuffer.get())
        }

        val sortedOutput = outputs.sortedDescending()
        val index = outputs.indexOf(sortedOutput[0])

        Log.d("TAG", "translate: $index")
    }

}
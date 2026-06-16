
platform :ios, '15.0'

target 'FormAI' do
  use_frameworks!
  pod 'MediaPipeTasksVision'
  pod 'SVProgressHUD'
end

# MediaPipe's pods don't ship with a high deployment target; keep CocoaPods
# from overriding the app's settings and silence the min-version churn.
post_install do |installer|
  installer.pods_project.targets.each do |t|
    t.build_configurations.each do |config|
      config.build_settings['IPHONEOS_DEPLOYMENT_TARGET'] = '15.0'
    end
  end
end

# FacialExpressionBased-on-Hierarchical-DNN
Efficient Facial Expression Recognition Algorithm Based on Hierarchical Deep Neural Network Structure

Paper link: IEEE Access, Vol. 7, 2019, https://ieeexplore.ieee.org/abstract/document/8673885

# Configuration Setting
With fer_env.yml
  1. Open it as a file text file, change the bottom prefix:~ part according to the path and name of the anaconda you installed, and save it.
  2. Execute with the administrator authority of anaconda prompt.
  3. Move to the path with the yml file.
  4. Run "conda env create -f fer_env.yml"
  5. Check the virtual environment setting after executing "conda activate ferrenv".

※If you get SystemError: execution of module h5py.utils raised unreported exception, run "pip install spladder".

# Model test
  1. Run "fer_main_6.py".
  2. Select dataset you want to test as from 1 to 7: 
  
  ::
  
    ----------------------------------------
    >>          FER DEMO SYSTEM            <<
    ----------------------------------------- 
    > (1) : New Image
    > (2) : CK+
    > (3) : JAFFE
    > (4) : FERG
    > (5) : AffectNet
    > (6) : ALL100
    > (7) : EXIT
    
  ::
  You can slect on job.
  
# Model training
   1. Training with LBP and CNN model
      - Run "train_app_6.py" and you can see the selection of datasets as:

  ::
  
    ----------------------------------------
    >>          FER TRAIN SYSTEM              <<
    ----------------------------------------- 
    > (0) : FERG
    > (1) : JAFFE
    > (2) : CK+
    > (3) : AffectNet
    > (4) : EXIT
       
  ::
  
     - You can find the saved model as "app_cnn_6/model/{iteration value}/[dataset name]_app_cnn~.hdf5"  (see lines 4811~4871 in "train_app_6.py")
   
   2. Training for Geometric CNN model

    - Run "pairwise_classifier_new.py".
    - Trained result: each pairwise model saved in "geo_cnn_6/" folder.
    
    ※ You can use "autoencoder" when you need to generate some neutral image in Affectnet dataset.  You can run "Autoencoder_main.py" to do this.
    
    3. Combined HDNN FER results
    
     You can run "fer_main_6.py" as Model Test procedure.

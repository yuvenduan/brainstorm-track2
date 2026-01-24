 1. export_video.py - Major Updates                                                                                      
                                                                                                                          
  Bad Channel Detection:                                                                                                  
  - Now uses first 1000 frames (configurable via --detection-frames)                                                      
  - Channels that don't change over time are detected via variance analysis                                               
  - Imputation automatically skipped if no bad channels found                                                             
                                                                                                                          
  Maximum Intensity Projection (MIP):                                                                                     
  - Added apply_mip() function for smoothing with overlapping windows                                                     
  - Default: 50-frame windows with 25-frame stride (50% overlap)                                                          
  - Configurable via --mip-window and --mip-stride                                                                        
                                                                                                                          
  Frame Range Annotation:                                                                                                 
  - Added render_frames() function that overlays frame ranges                                                             
  - Displays sample range (e.g., "0-17") in top right corner                                                              
  - White text on black background for readability                                                                        
                                                                                                                          
  New Processing Pipeline:                                                                                                
  generate_frames() → apply_mip() → render_frames() → export_video()                                                      
    (standardize)      (max proj)      (colormap +      (write MP4)                                                       
                                        text overlay)                                                                     
                                                                                                                          
  New CLI Parameters:                                                                                                     
  - --detection-frames 1000 - frames for bad channel detection                                                            
  - --mip-window 50 - MIP window size                                                                                     
  - --mip-stride 25 - MIP stride (half-overlap)          

    Usage Examples                                                                                                          
                                                                                                                          
  # Generate deliverables                                                                                                 
  uv run python scripts/export_video.py -d super_easy -o /store1/candy/bci/super_easy.mp4 --verbose                       
  uv run python scripts/export_video.py -d hard -o /store1/candy/bci/hard.mp4 --verbose                                   
                                                                                                                          
  # Custom MIP smoothing                                                                                                  
  uv run python scripts/export_video.py -d hard -o /store1/candy/bci/hard_smooth.mp4 --mip-window 100 --mip-stride 50     
                                                                                                                          
  # Adjust bad channel detection sensitivity                                                                              
  uv run python scripts/export_video.py -d super_easy -o /store1/candy/bci/super_easy.mp4 --variance-threshold 1e-7       
                                                                                                                          
  The script is ready to generate the required MP4 videos with frame range annotations and MIP smoothing!   


    The pipeline now includes: bad channels → interpolation → artifact removal → frequency decomposition →  per-band MIP → 4-panel visualization. 

     1. Test bad channel stuck detection:                                                                     
 # Should report skipping imputation if >75% stuck                                                        
 python scripts/export_video.py -d super_easy -o /tmp/test.mp4 --verbose                                  
                                                                                                          
 2. Test artifact removal:                                                                                
 # Should report 60Hz and white noise detection/removal in verbose                                        
 python scripts/export_video.py -d hard -o /tmp/test.mp4 --verbose                                        
 python scripts/export_video.py -d hard -o /tmp/test.mp4 --skip-artifact-removal --verbose                
                                                                                                          
 3. Test multi-panel output:                                                                              
 # Verify 4-panel layout with colorbar                                                                    
 python scripts/export_video.py -d super_easy -o /store1/candy/bci/super_easy.mp4 --verbose               
 python scripts/export_video.py -d hard -o /store1/candy/bci/hard.mp4 --verbose                           
                                                                                                          
 4. Manual inspection:                                                                                    
 - Open output MP4 files                                                                                  
 - Verify 4 panels (2x2 grid) showing different frequency bands                                           
 - Verify colorbar on right side with amplitude labels                                                    
 - Verify frame range text in top-right corner                                                            
 - Verify band labels on each panel                                                                       
                                                                                                          
 Processing Pipeline After Changes                                                                        
                                                                                                          
 Load parquet (150000 x 1024)                                                                             
          |                                                                                               
 Detect bad channels (first 1000 frames)                                                                  
   - Check stuck value (>75% same) -> skip if true                                                        
   - Modified std: only non-mean values                                                                   
          |                                                                                               
 Impute bad channels (if any, not skipped)                                                                
          |                                                                                               
 Remove artifacts (if not --skip-artifact-removal)                                                        
   - 60Hz notch filter                                                                                    
   - White noise reduction                                                                                
          |                                                                                               
 Frequency decomposition -> 4 arrays                                                                      
   - Theta/Alpha (4-12Hz)                                                                                 
   - Beta (12-30Hz)                                                                                       
   - Low Gamma (30-70Hz)                                                                                  
   - High Gamma (70-150Hz)                                                                                
          |                                                                                               
 Per-band normalization (first 100 frames each)                                                           
          |                                                                                               
 Generate frames per band -> MIP per band -> Render 2x2 + colorbar                                        
          |                                                                                               
 Export to MP4                                       

   uv run python scripts/export_video.py -d super_easy -o /store1/candy/bci/super_easy.mp4 --verbose       
                                                                         
                                                                                                          
  Full options available:                                                                                 
  uv run python scripts/export_video.py \                                                                 
    -d hard \                                                                                             
    -o /store1/candy/bci/hard.mp4 \                                                                       
    --fps 30.0 \                                                                                          
    --frame-size 512 \                                                                                    
    --detection-frames 1000 \                                                                             
    --stats-frames 1000 \                                                                                 
    --mip-window 500 \                                                                                    
    --mip-stride 250 \                                                                                    
    --n-std 3.0 \                                                                                         
    --verbose  


      Open your browser and go to:                                                                            
  http://localhost:8000                                                                                   
                                                                                                          
  The app is already configured to connect to ws://localhost:8766 (the processor).                        
                                                                                                          
  Just click the Connect button and you should see the processed neural data stream with:                 
  - ✅ 4 bad channels ([51, 232, 419, 582]) being imputed                                                 
  - ✅ 3x3 spatial smoothing applied                                                                      
  - ✅ Full temporal resolution (no time averaging)                                                       
                                                                                                          
  Current Setup                                                                                           
                                                                                                          
  Data Stream (ws://localhost:8765)                                                                       
           ↓                                                                                              
      Processor (ws://localhost:8766)                                                                     
      - Detects & fixes 4 bad channels                                                                    
      - Applies 3x3 median filter                                                                         
           ↓                                                                                              
      Web App (http://localhost:8000)                


  Commit Summary                                                                                          
                                                                                                          
  Commit: a1272a2                                                                                         
  Branch: candy                                                                                           
  Status: Pushed to origin                                                                                
                                                                                                          
  Files Changed:                                                                                          
                                                                                                          
  - ✅ scripts/process_stream.py (new) - Real-time processing pipeline                                    
  - ✅ docs/realtime_processing.md (new) - Documentation                                                  
  - ✅ example_app/app.js (modified) - Updated colormap range to [-0.2, +0.25]                            
  - ✅ example_app/index.html (modified) - Updated server URL to port 8766 and colorbar labels            
                                                                                                          
  What Was Committed:                                                                                     
                                                                                                          
  Real-time high gamma (70-150Hz) processing pipeline with:                                               
  - Bad channel detection and imputation                                                                  
  - Bandpass filtering with Hilbert envelope extraction                                                   
  - Standardization and normalization                                                                     
  - Min-max scaling for full colormap utilization                                                         
  - Spatial smoothing with 3x3 median filter                                                              
  - Stateful causal filtering for continuous processing                                                   
                                                                                                          
  The processor is now live and streaming processed data from port 8766!                                                       
                                                                                                    
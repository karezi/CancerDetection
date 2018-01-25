#import sys
#sys.path.append('/opt/ASAP/bin')

import multiresolutionimageinterface as mir

reader = mir.MultiResolutionImageReader()
mr_image = reader.open('/home/suidong/Documents/camelyon17_data_backup/slide/patient_000_node_0.tif')
annotation_list = mir.AnnotationList()
xml_repository = mir.XmlRepository(annotation_list)
xml_repository.setSource('/home/suidong/Documents/camelyon17_data_backup/lesion_annotations/patient_000_node_0.xml')
xml_repository.load()
annotation_mask = mir.AnnotationToMask()
label_map = {'metastases': 255, 'normal': 0}
output_path = '/home/suidong/Documents/camelyon17_data_backup/test/test.tif'
annotation_mask.convert(annotation_list, output_path, mr_image.getDimensions(), mr_image.getSpacing(), label_map)

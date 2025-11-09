from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
	'ffhq_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['ffhq'],
		'train_target_root': dataset_paths['ffhq'],
		'test_source_root': dataset_paths['celeba_test'],
		'test_target_root': dataset_paths['celeba_test'],
	},
 	'frontalization': {
		'transforms': transforms_config.FrontalizationTransforms,
		'train_source_root': dataset_paths['mh_dataset'],
		'train_target_root': dataset_paths['mh_dataset'],
		'test_source_root': dataset_paths['mh_dataset'],
		'test_target_root': dataset_paths['mh_dataset'],
	}
}

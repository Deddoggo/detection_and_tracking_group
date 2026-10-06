import copy
import unittest
import xml.etree.ElementTree as ET

from diagnostics.cvat_annotations import REVIEW_FIELDS, convert, qa


def mapping(count=4):
    return {'clip':{'frame_count':count,'width':100,'height':80},
            'source':{'video':'synthetic','start_frame':832,'end_frame_exclusive':832+count},
            'frames':[{'clip_frame':i,'mot_frame':i+1,'source_frame':832+i,
                       'source_timestamp_seconds':30+i*0.04} for i in range(count)],
            'verification':{'all_decoded_pixel_hashes_match_source':True}}


def xml(body,count=4):
    return ET.fromstring(f'<annotations><version>1.1</version><meta><task><size>{count}</size>'
                         '<flipped>False</flipped><original_size><width>100</width><height>80</height>'
                         '</original_size></task></meta>'+body+'</annotations>')


def box(n,x=0,outside=0,uncertain=False,pid=1):
    return f'<box frame="{n}" xtl="{x}" ytl="0" xbr="{x+10}" ybr="20" outside="{outside}" occluded="0" keyframe="1">'\
           f'<attribute name="person_id">{pid}</attribute><attribute name="identity_uncertain">{str(uncertain).lower()}</attribute></box>'


class CvatTests(unittest.TestCase):
    def test_linear_interpolation_maps_to_source_and_stays_pending(self):
        d=convert(xml('<track id="0" label="person">'+box(0)+box(3,30)+'</track>'),mapping())
        self.assertEqual([f['frame'] for f in d['frames']],[832,833,834,835])
        self.assertEqual([f['items'][0]['bbox'][0] for f in d['frames']],[0,10,20,30])
        self.assertEqual(d['annotation_status'],'pending')
        self.assertFalse(qa(d)['tracking_export_ready'])

    def test_outside_gap_reappearance_and_uncertainty(self):
        d=convert(xml('<track id="0" label="person">'+box(0)+box(1,outside=1)+box(3,uncertain=True)+'</track>'),mapping())
        self.assertEqual([len(f['items']) for f in d['frames']],[1,0,0,1])
        self.assertIsNone(d['frames'][3]['items'][0]['person_id'])
        self.assertEqual(d['frames'][3]['items'][0]['candidate_person_id'],1)

    def test_explicit_human_review_is_required(self):
        review={k:True for k in REVIEW_FIELDS};review.update(reviewer='synthetic tester',review_date='2026-10-05')
        d=convert(xml('<track id="0" label="person">'+box(0)+'</track>'),mapping(),review)
        self.assertEqual(d['annotation_status'],'human_reviewed')
        self.assertTrue(qa(d)['tracking_export_ready'])

    def test_duplicate_person_ids_and_geometry_fail(self):
        body=''.join('<track id="'+str(n)+'" label="person">'+box(0)+'</track>' for n in range(2))
        self.assertRaises(ValueError,convert,xml(body),mapping())
        self.assertRaises(ValueError,convert,xml('<track id="0" label="person">'+box(0,95)+'</track>'),mapping())

    def test_image_export_requires_every_frame_and_track_identity(self):
        images=''.join(f'<image id="{n}" width="100" height="80">'
                       '<box label="person" xtl="0" ytl="0" xbr="10" ybr="20" occluded="0">'
                       '<attribute name="person_id">7</attribute><attribute name="track_id">0</attribute>'
                       '</box></image>' for n in range(4))
        d=convert(xml(images),mapping());self.assertEqual(d['frames'][2]['items'][0]['person_id'],7)
        self.assertRaises(ValueError,convert,xml(images.replace('id="3"','id="2"')),mapping())
        self.assertRaises(ValueError,convert,xml(images.replace('<attribute name="track_id">0</attribute>','')),mapping())

    def test_step_and_flip_rejected(self):
        root=xml('<track id="0" label="person">'+box(0)+'</track>')
        ET.SubElement(root.find('meta/task'),'frame_filter').text='step=2'
        self.assertRaises(ValueError,convert,root,mapping())
        root=xml('');root.find('meta/task/flipped').text='True'
        self.assertRaises(ValueError,convert,root,mapping())


if __name__=='__main__':unittest.main()
